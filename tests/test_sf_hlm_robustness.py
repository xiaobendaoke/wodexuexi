"""Registration, timing and frozen physical-variant regression; run remotely."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from paths import REPO_ROOT

ENTRY = Path(os.environ.get('SF_HLM_ROBUSTNESS_ENTRY',
    REPO_ROOT / 'scripts/analysis/sf_hlm_robustness.py'))
spec = importlib.util.spec_from_file_location('robustness_runner', ENTRY)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class RegistrationTests(unittest.TestCase):
    def setUp(self):
        self.tasks = m.registered_tasks(list(range(10)), list(range(5)), 6,
            [0, 249, 499, 749], capture_episodes=2)

    def test_population_is_fixed_and_nominal_scenario_is_reused(self):
        counts = {p: sum(t[0] == p for t in self.tasks) for p in m.PHASES}
        self.assertEqual(counts, dict(capture=100, robustness=14400, benchmark=400))
        self.assertEqual(len(set(self.tasks)), 14900)
        self.assertFalse(any(t[0] == 'robustness' and t[1] == m.NOMINAL for t in self.tasks))

    def test_timing_waits_for_every_shared_job(self):
        shared = {t for t in self.tasks if t[0] != 'benchmark'}
        missing = next(iter(shared))
        self.assertFalse(any(t[0] == 'benchmark' for t in m.eligible(self.tasks, shared-{missing}, set())))
        self.assertEqual(len(m.eligible(self.tasks, shared, set())), 400)

    def test_timing_dispatch_never_queues_more_than_reserved_cores(self):
        jobs = [t for t in self.tasks if t[0] == 'benchmark']
        self.assertEqual(m.submissions(jobs[6:], jobs[:6], 192, 8), jobs[6:8])
        self.assertEqual(m.submissions(jobs[8:], jobs[:8], 192, 8), [])
        shared = [t for t in self.tasks if t[0] == 'robustness']
        self.assertEqual(len(m.submissions(shared, [], 192, 8)), 192)

    def test_active_jobs_are_not_reissued(self):
        job = self.tasks[0]
        self.assertNotIn(job, m.eligible(self.tasks, set(), {job}))

    def test_decode_restores_numeric_action_keys(self):
        rows = [{'options': {'0': {'target': 0}, '1': {'target': 'mbs'}}}]
        decoded = m.decode_rows(rows)
        self.assertEqual(set(decoded[0]['options']), {0, 1})
        self.assertEqual(rows[0]['options'], {'0': {'target': 0}, '1': {'target': 'mbs'}})

    def test_wall_condition_keeps_frozen_call_budget(self):
        self.assertEqual(m.conditions(), [('free', None), ('fixed_mbs', None),
            ('wall', .02), ('wall', .05), ('wall', .1)])
        self.assertEqual(m.MATCHING_BUDGET, 16)
        with self.assertRaises(ValueError): m.remaining_wall(0, 0, 0)
        self.assertEqual(m.remaining_wall(.02, 1., 1.03), 0.)
        self.assertAlmostEqual(m.remaining_wall(.05, 1., 1.02), .03)

    def test_receipt_checks_identity_and_raw_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = argparse.Namespace(results_root=Path(tmp))
            task = next(t for t in self.tasks if t[0] == 'robustness')
            folder = m.folder(args, task); folder.mkdir(parents=True)
            (folder/'episode.json').write_text('{}')
            protocol = {'steps': 1000, 'parent_protocol_sha256': 'parent',
                'source_hash': 'source', 'snapshot_steps': [0, 249, 499, 749]}
            (args.results_root/'protocol.json').write_text('{}')
            contract = m.task_contract(args, protocol, task)
            data = {'status': 'PASS', 'contract': contract, 'steps': 1000,
                'files': {'episode.json': m.sha(folder/'episode.json')}}
            (folder/'result.json').write_text(json.dumps(data))
            self.assertTrue(m.verify(args, protocol, task))
            (folder/'episode.json').write_text('changed')
            with self.assertRaises(ValueError):m.verify(args, protocol, task)

    def test_partial_episode_receipt_cannot_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = argparse.Namespace(results_root=Path(tmp))
            task = next(t for t in self.tasks if t[0] == 'robustness')
            folder = m.folder(args, task); folder.mkdir(parents=True)
            protocol = {'steps': 1000, 'parent_protocol_sha256': 'parent',
                'source_hash': 'source', 'snapshot_steps': [0, 249, 499, 749]}
            (args.results_root/'protocol.json').write_text('{}')
            (folder/'result.json').write_text(json.dumps(dict(status='PASS',
                contract=m.task_contract(args, protocol, task), steps=8, files={})))
            with self.assertRaises(ValueError):m.verify(args, protocol, task)

    def test_protocol_serialization_can_resume_without_changing_tasks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);parent=root/'parent';release=root/'release'
            (parent/'analysis').mkdir(parents=True)
            (parent/'protocol.json').write_text(json.dumps(dict(smoke=False,
                source_hash='source',confirmation_blocks=list(range(10)),
                confirmation_workloads=list(range(5)),snapshot_steps=[0,250,500,750])))
            (parent/'status.json').write_text('{"phase":"CONFIRMATION_COMPLETE"}')
            final=dict(groups={k:{} for k in m.METHODS},outcomes={'learned':{'primary_passed':True}})
            (parent/'analysis/confirmation.json').write_text(json.dumps(final))
            (parent/'analysis/confirmation_eval.json').write_text('{}')
            for name in m.FILES:
                p=release/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('frozen')
            args=argparse.Namespace(confirmation_root=parent,release_root=release,smoke=False)
            protocol=m.make_protocol(args)
            self.assertEqual(json.loads(json.dumps(protocol)),protocol)
            self.assertEqual(len(protocol['tasks']),14900)

    def test_duplicate_live_workers_are_rejected_and_single_owner_adopted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);proc=root/'proc';proc.mkdir()
            args=argparse.Namespace(**{k:root/k for k in m.PATH_ARGS})
            task=self.tasks[0];words=m.command(args,task)
            def process(pid):
                path=proc/str(pid);path.mkdir();(path/'cmdline').write_bytes(('\0'.join(words)+'\0').encode())
            process(11);res=SimpleNamespace(process_identity=lambda pid:str(pid*10))
            self.assertEqual(m.live_workers(args,self.tasks,res,proc),{task:(11,'110')})
            process(12)
            with self.assertRaises(ValueError):m.live_workers(args,self.tasks,res,proc)

    def test_parent_algorithm_and_analysis_files_are_frozen(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);args=argparse.Namespace(**{k:root/k for k in m.PATH_ARGS})
            for key in m.PATH_ARGS:getattr(args,key).mkdir()
            parent={}
            for key,location in [('files','followup_root'),('extension_files','extension_root'),('adaptation_files','adaptation_root')]:
                f=getattr(args,location)/'frozen.py';f.write_text('original')
                parent[key]={'frozen.py':m.sha(f)}
            (args.confirmation_root/'protocol.json').write_text(json.dumps(parent))
            (args.confirmation_root/'analysis').mkdir()
            p={'parent_protocol_sha256':m.sha(args.confirmation_root/'protocol.json'),'files':{}}
            for name,key in [('confirmation_eval.json','parent_quality_sha256'),('confirmation.json','parent_final_sha256')]:
                f=args.confirmation_root/'analysis'/name;f.write_text('{}');p[key]=m.sha(f)
            m.verify_releases(args,p)
            (args.followup_root/'frozen.py').write_text('changed')
            with self.assertRaises(ValueError):m.verify_releases(args,p)


@unittest.skipUnless(os.environ.get('SF_HLM_ROBUSTNESS_NUMERICAL'), 'Remote numerical regressions')
class NumericalTests(unittest.TestCase):
    def test_upper_capacity_and_mbs_are_preserved_for_every_variant(self):
        import numpy as np
        import config
        from utils.logger import refresh_derived_config_fields
        initial = {k: getattr(config, k) for k in ('NUM_UES', 'MAX_ASSOCIATED_UES',
            'OBS_DIM_SINGLE', 'UAV_COMPUTING_CAPACITY', 'MBS_COMPUTING_CAPACITY')}
        for variant in m.VARIANTS.values():
            for k,v in initial.items():setattr(config,k,v.copy() if hasattr(v,'copy') else v)
            m.apply_variant(variant)
            self.assertEqual(config.MAX_ASSOCIATED_UES, initial['MAX_ASSOCIATED_UES'])
            self.assertEqual(config.OBS_DIM_SINGLE, initial['OBS_DIM_SINGLE'])
            self.assertEqual(config.MBS_COMPUTING_CAPACITY, initial['MBS_COMPUTING_CAPACITY'])
            self.assertEqual(config.MAX_OFFLOAD_REQUESTS_PER_UAV, variant['ue_count'])
            np.testing.assert_array_equal(config.UAV_COMPUTING_CAPACITY,
                initial['UAV_COMPUTING_CAPACITY']*variant['compute_scale'])
        for k,v in initial.items():setattr(config,k,v)
        refresh_derived_config_fields()

    def test_wall_budget_includes_input_and_preserves_policy_call_limit(self):
        import numpy as np
        rows=[dict(ue_id=7,source=0,options={0:{'target':0}})]
        reference=dict(energy=2.,misses=0)
        diagnostic=dict(matching_calls=1,reference_predicted=reference,
            chosen_predicted=reference,common_energy_cap=2.1)
        search=Mock(return_value=((0,),diagnostic,[]))
        fixed=Mock(return_value=((0,),dict(diagnostic,fixed_mbs_count=0),[]))
        modules={
            'utils.deadline_repair_online':SimpleNamespace(greedy_assignment=lambda rows:((0,),0)),
            'utils.deadline_repair':SimpleNamespace(assess=lambda rows,a:reference),
            'utils.sf_hlm_matching':SimpleNamespace(feedback=lambda *a,**k:np.zeros((6,6))),
            'utils.sf_hlm_v2':SimpleNamespace(search=search,validate_metadata=lambda rows:None),
            'utils.sf_hlm_followup':SimpleNamespace(fixed_mbs_search=fixed),
            'marl_models.sf_hlm_v2':SimpleNamespace(service_view=lambda *a:None)}
        r=SimpleNamespace(base=SimpleNamespace(prepare=lambda env:(None,None,[(0,0,7)],rows),
            actions_from_assignment=lambda *a:np.zeros((5,1)),serial=lambda value:value))
        models={'upper':SimpleNamespace(select_actions=lambda *a,**k:np.zeros((5,2)))}
        with patch.dict('sys.modules',modules):
            with patch.object(m.time,'perf_counter',side_effect=[1.,1.03,1.04]):
                out=m.timed_policy(None,None,'learned',models,None,None,r,wall=.02)
            self.assertEqual(out[2]['diagnostic']['matching_calls'],0)
            self.assertAlmostEqual(out[2]['policy_seconds'],.04)
            search.assert_not_called()
            with patch.object(m.time,'perf_counter',side_effect=[1.,1.01,1.025]):
                m.timed_policy(None,None,'learned',models,None,None,r,wall=.05)
            self.assertEqual(search.call_args.kwargs['budget'],16)
            self.assertAlmostEqual(search.call_args.kwargs['wall_seconds'],.04)
            with patch.object(m.time,'perf_counter',side_effect=[1.,1.025]):
                m.timed_policy(None,None,'learned',models,None,None,r,fixed=True)
            self.assertEqual(fixed.call_args.kwargs['budget'],16)


if __name__ == '__main__': unittest.main()
