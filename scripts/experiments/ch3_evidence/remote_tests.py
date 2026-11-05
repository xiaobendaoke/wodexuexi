"""Server-side behavioral checks for evidence integrity, observation isolation and metrics."""
import copy
import gzip
import json
from pathlib import Path
import tempfile
import unittest
from argparse import Namespace
import numpy as np
import torch
import config
from environment.env import Env as BaseEnv
from ch3_runtime import ExperimentEnv,configure,packet_kept,compatible
from utils.canonical_experiment import config_snapshot,digest
from run_canonical_experiment import configure_method
from ch3_analysis import effect,diagnostic_pair,verify_job


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        import importlib
        importlib.reload(config)
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.args=Namespace(ch3_actor_width=128,ch3_critic_width=128,ch3_uniform=False,ch3_demand_radius=460.,
            ch3_drop=0.,ch3_delay=0,ch3_record_requests=False,output=self.tmp.name,ch3_boundary=False,method='lower_no_attention')
        configure(self.args);torch.set_num_threads(1)

    def test_registered_cells_same_information_and_lower(self):
        groups={}
        for method in ('lower_no_attention','lower_mlp_upper_uncovered_obs','no_attention','vanilla_lower_mlp_upper_uncovered_obs'):
            np.random.seed(21103401);torch.manual_seed(21103401)
            models=configure_method(method)
            groups[method]=(type(models['upper']).__name__,type(models['lower']).__name__,config.UPPER_UE_OBSERVATION_MODE)
        self.assertEqual(len({r[1] for r in groups.values()}),1)
        self.assertEqual(groups['lower_no_attention'][2],groups['no_attention'][2])
        self.assertEqual(groups['lower_mlp_upper_uncovered_obs'][2],groups['vanilla_lower_mlp_upper_uncovered_obs'][2])

    def test_default_observation_and_transition_equivalence(self):
        for mode in ('covered','covered_plus_uncovered'):
            config.UPPER_UE_OBSERVATION_MODE=mode
            a,b=BaseEnv(21103501),ExperimentEnv(21103501)
            ao,bo=a.reset(),b.reset();np.testing.assert_array_equal(ao,bo)
            for _ in range(3):
                x=a.step(np.zeros((5,2)));y=b.step(np.zeros((5,2)))
                np.testing.assert_array_equal(x[0],y[0]);self.assertEqual(x[1:],y[1:])
                self.assertEqual(a._rng.get_state()[2:],b._rng.get_state()[2:])
                np.testing.assert_array_equal(a._rng.get_state()[1],b._rng.get_state()[1])

    def test_radius100_matches_covered_on_same_state(self):
        config.UPPER_UE_OBSERVATION_MODE='covered_plus_uncovered'
        a=ExperimentEnv(21103501);a.reset();clone=copy.deepcopy(a)
        config.CH3_DEMAND_RADIUS=100.
        expanded=a._get_obs();config.UPPER_UE_OBSERVATION_MODE='covered'
        covered=clone._get_obs();np.testing.assert_array_equal(expanded,covered)

    def test_missing_is_deterministic_and_rng_free(self):
        state=np.random.get_state()
        first=[packet_kept(1,2,3,4,u,.5) for u in range(100)]
        self.assertEqual(first,[packet_kept(1,2,3,4,u,.5) for u in range(100)])
        self.assertTrue(any(first));self.assertFalse(all(first))
        np.testing.assert_array_equal(state[1],np.random.get_state()[1])
        self.assertTrue(packet_kept(1,2,3,4,5,0));self.assertFalse(packet_kept(1,2,3,4,5,1))

    def test_delay_no_warmup_refill_and_stale_metadata(self):
        config.UPPER_UE_OBSERVATION_MODE='covered_plus_uncovered';config.CH3_DELAY=1
        env=ExperimentEnv(21103501);obs=env.reset()
        for u,row in zip(env.uavs,obs):
            own=min(len(u.current_covered_ues),30)
            np.testing.assert_array_equal(row[85:].reshape(30,8)[own:],0)
        historic=copy.deepcopy(env._telemetry[-1][1]);obs=env.step(np.zeros((5,2)))[0]
        for u,row in zip(env.uavs,obs):
            own=min(len(u.current_covered_ues),30);slots=row[85:].reshape(30,8)[own:]
            own_ids={ue.id for ue in u.current_covered_ues}
            received=[packet for packet in historic[u.id] if packet[0] not in own_ids]
            self.assertEqual(env.ch3_slot_counts[u.id]['visible_uncovered'],len(received))
            self.assertEqual(env.ch3_slot_counts[u.id]['truncated'],max(len(u.current_covered_ues)+len(received)-30,0))
            old=[np.asarray(packet[2],dtype=np.float32) for packet in historic[u.id]]
            for slot in slots:
                if np.any(slot):self.assertTrue(any(np.array_equal(slot[2:],x) for x in old))
        self.assertEqual(env._telemetry[0][0][1],0)

    def test_frozen_interfaces_under_ue_changes(self):
        for n in (60,140):
            config.NUM_UES=n;configure(self.args)
            self.assertEqual(config.OBS_DIM_SINGLE,325);self.assertEqual(config.MAX_OFFLOAD_REQUESTS_PER_UAV,100)
            env=ExperimentEnv(21103501);self.assertEqual(np.asarray(env.reset()).shape,(5,325))
            self.assertEqual(env.get_offloading_obs_and_masks()[0].shape[-1],1405)

    def test_lower_overflow_is_rejected(self):
        env=ExperimentEnv(21103501);env.reset()
        class Request:is_service=True
        class User:current_request=Request()
        env.uavs[0]._current_covered_ues=[User() for _ in range(101)]
        with self.assertRaisesRegex(RuntimeError,'INTERFACE_NOT_APPLICABLE'):env.get_offloading_obs_and_masks()

    def test_diagnostic_is_passive_and_counts_complete(self):
        config.UPPER_UE_OBSERVATION_MODE='covered_plus_uncovered'
        ExperimentEnv.record_requests=False
        a=ExperimentEnv(21103501);a.reset()
        baseline=a.step(np.zeros((5,2)))
        self.args.ch3_record_requests=True;configure(self.args)
        b=ExperimentEnv(21103501);b.reset()
        observed=b.step(np.zeros((5,2)))
        np.testing.assert_array_equal(baseline[0],observed[0]);self.assertEqual(baseline[1:],observed[1:])
        path=next((Path(self.tmp.name)/'requests').glob('episode_*.gz'))
        with gzip.open(path,'rt') as f:rows=[json.loads(x) for x in f]
        self.assertEqual(len(rows),observed[2]['generated_service_count'])
        for row in rows:
            if row['admitted']:
                self.assertAlmostEqual(row['latency'],sum(row[k] for k in ('upload_latency','forward_latency','fetch_latency','compute_latency')),places=7)

    def test_checkpoint_boundary_allowlist(self):
        s=config_snapshot();c=dict(method=self.args.method,source_hash='source',config_hash=digest(s),config=s)
        self.assertTrue(compatible(c,s,'source',self.args))
        changed=dict(s,NUM_UES=140);self.assertFalse(compatible(c,changed,'source',self.args))
        self.args.ch3_boundary=True;self.assertTrue(compatible(c,changed,'source',self.args))
        changed['REWARD_W_DSR']=100;self.assertFalse(compatible(c,changed,'source',self.args))
        self.assertFalse(compatible(c,s,'wrong_source',self.args))

    def test_upper_width_changes_do_not_change_lower(self):
        models=configure_method('no_attention');count=sum(p.numel() for p in models['lower'].actor.parameters())
        self.args.ch3_actor_width=51;self.args.ch3_critic_width=70;configure(self.args)
        other=configure_method('no_attention')
        self.assertEqual(count,sum(p.numel() for p in other['lower'].actor.parameters()))
        self.assertNotEqual(sum(p.numel() for p in models['upper'].actor.parameters()),sum(p.numel() for p in other['upper'].actor.parameters()))

    def test_inference_training_seed_unit_and_partial_rejection(self):
        plan=dict(bootstrap_seed=21103601,bootstrap_draws=1000)
        result=effect(np.ones((10,5))*.03,plan)
        self.assertAlmostEqual(result['mean_difference'],.03);self.assertTrue(result['criterion_met'])
        self.assertEqual(len(result['seed_mean_differences']),10)
        with self.assertRaises(ValueError):effect([[1]],plan)
        with self.assertRaises(FileNotFoundError):verify_job(Path(self.tmp.name),'x',6,1000)

    def test_diagnostics_distinguish_non_nested_sets(self):
        root=Path(self.tmp.name)
        for side,admitted in [('left',[True,False,True]),('right',[True,True,False])]:
            folder=root/side/'requests';folder.mkdir(parents=True)
            with gzip.open(folder/'episode_00000.jsonl.gz','wt') as f:
                for u,flag in enumerate(admitted):
                    row=dict(workload_seed=1,episode=0,slot=0,ue=u,request_id=0,size=2,deadline=1,priority=1,cycles=4,admitted=flag,success=flag)
                    f.write(json.dumps(row)+'\n')
        result=diagnostic_pair(root/'left',root/'right')
        self.assertEqual(result['both_admitted']['count'],1)
        self.assertEqual(result['left_only_admitted']['count'],1)
        self.assertEqual(result['right_only_admitted']['count'],1)


if __name__=='__main__':unittest.main()
