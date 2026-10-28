"""Numerical C0 boundary tests, loaded only after frozen server bootstrap."""
import argparse
import copy
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/analysis'))
import chapter5_stage_b as b


class PrefixTests(unittest.TestCase):
    def test_registered_budgets(self):
        from chapter5_budget_prefix import PrefixPolicy
        with FAST.FastRuntime() as rt:
            with self.assertRaises(ValueError):PrefixPolicy(rt,PRICES,PRICES['powers'],1)

    def test_empty_and_g(self):
        from chapter5_budget_prefix import PrefixPolicy,NativeRecordedPolicy
        from utils.sf_hlm_priced_runtime import PricedPolicy
        with FAST.FastRuntime() as rt:
            native=NativeRecordedPolicy(rt,None,PRICES,PRICES['powers'])
            expected=native.run([],num_uavs=5,kind='mixture_price',cap=10,trace=True)
            for budget in [2,4,6,10]:
                impl=PrefixPolicy(rt,PRICES,PRICES['powers'],budget)
                got=impl.run([],num_uavs=5,kind='mixture_price',cap=budget,trace=True)
                self.assertEqual(got[1]['matching_calls'],0)
                self.assertEqual(got[0],expected[0])
            impl=PrefixPolicy(rt,PRICES,PRICES['powers'],0)
            left=impl.run(ROWS,num_uavs=5,kind='mixture_price',cap=0,trace=True)
            right=PricedPolicy(rt,None,PRICES,PRICES['powers']).run(ROWS,num_uavs=5,kind='native_G',trace=True)
            self.assertEqual(b.normal(left),b.normal(right))
            self.assertEqual(impl.last_trace['initialization_calls'],0)

    def test_failed_matching_counts_and_no_future_matrices(self):
        from chapter5_budget_prefix import PrefixPolicy
        from utils import sf_hlm_calibrated_matching as cm
        from utils.sf_hlm_cache_mixture import CacheMixtureKernel
        original=CacheMixtureKernel.probabilities;seen=[]
        def record(kernel,quotas):
            seen.extend(tuple(q) for q in quotas)
            return original(kernel,quotas)
        with FAST.FastRuntime() as rt,patch.object(cm,'solve',return_value=None),patch.object(CacheMixtureKernel,'probabilities',record):
            impl=PrefixPolicy(rt,PRICES,PRICES['powers'],4)
            action,diag,candidates=impl.run(ROWS,num_uavs=5,kind='mixture_price',cap=4,trace=True)
            visited=impl.last_trace['visited']
            self.assertEqual(diag['matching_calls'],1+len(visited))
            self.assertTrue(all(z['rejection']=='infeasible' for z in visited))
            self.assertEqual(len(candidates),1)
            self.assertEqual(candidates[0]['action'],-1)
            self.assertEqual(set(seen),{tuple(z['quota']) for z in visited})
            self.assertLessEqual(len(seen),3)

    def test_energy_rejections_keep_g(self):
        from chapter5_budget_prefix import PrefixPolicy
        from utils import sf_hlm_calibrated_matching as cm
        original=cm.solve
        def reject(*args,**kwargs):
            result=original(*args,**kwargs)
            if result is not None:
                result=copy.deepcopy(result);result['predicted']['energy']=1e100
            return result
        with FAST.FastRuntime() as rt,patch.object(cm,'solve',reject):
            impl=PrefixPolicy(rt,PRICES,PRICES['powers'],6)
            action,diag,candidates=impl.run(ROWS,num_uavs=5,kind='mixture_price',cap=6,trace=True)
            self.assertEqual(diag['matching_calls'],1+len(impl.last_trace['visited']))
            self.assertEqual(len(candidates),1)
            self.assertEqual(tuple(action),tuple(candidates[0]['assignment']))

    def test_full_call_b10_equivalence(self):
        import numpy as np
        from chapter5_budget_prefix import PrefixPolicy,NativeRecordedPolicy
        from utils.sf_hlm_software_delay import SoftwareServiceDelay
        _,_,runner,models,_,_=RUNTIME
        blob=(SAMPLE/'observed.bin').read_bytes();outputs=[]
        for native in [True,False]:
            env,obs=__import__('pickle').loads(blob)
            with FAST.FastRuntime() as rt:
                impl=NativeRecordedPolicy(rt,None,PRICES,PRICES['powers']) if native else PrefixPolicy(rt,PRICES,PRICES['powers'],10)
                rt.patch(CORE,'truncated_search',impl.run)
                motion,actions,info,rows=rt.complete_policy(CORE,env,obs,dict(kind='mixture_price',cap=10,seed=123),models,None,runner)
                with SoftwareServiceDelay(env,'measured') as hook:
                    hook.arm(info['request_ids'],.0123);next_obs,rewards,metrics=env.step(motion,offloading_actions=actions);outcome=hook.finish()
                outputs.append((motion,actions,info,rows,next_obs,rewards,metrics,outcome))
        a,z=outputs
        self.assertTrue(np.array_equal(a[0],z[0]));self.assertTrue(np.array_equal(a[1],z[1]))
        self.assertEqual(a[2]['assignment'],z[2]['assignment']);self.assertEqual(a[2]['request_ids'],z[2]['request_ids'])
        self.assertEqual(b.normal(a[2]['diagnostic']),b.normal(z[2]['diagnostic']))
        self.assertEqual(b.normal(a[3:]),b.normal(z[3:]))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--archive',type=Path,required=True);parser.add_argument('--development',type=Path,required=True)
    args=parser.parse_args()
    _,RUNTIME,FAST,CORE,PRICES=b.boot(args,620001)
    SAMPLE=sorted((args.development/'tasks').glob('620001_*_center/G/samples/*'))[0]
    ROWS=b.read(SAMPLE/'public.json')['rows']
    for row in ROWS:row['options']={int(k):v for k,v in row['options'].items()}
    assert ROWS
    unittest.main(argv=[sys.argv[0]])
