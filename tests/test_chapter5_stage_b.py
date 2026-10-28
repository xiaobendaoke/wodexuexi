"""Targeted protocol and branch-intervention checks; execute on server."""
import argparse
import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/analysis'))
import chapter5_stage_b as runner


class StageB(unittest.TestCase):
    def test_preselected_strata(self):
        p=runner.read(OUTPUT/'protocol.json')
        self.assertEqual(len(p['samples']),30)
        for selected in p['samples'].values():
            self.assertEqual([x//25 for x in selected],list(range(40)))
        self.assertEqual(len(p['bundles']),75)
        self.assertEqual(sum(len(runner.methods(s)) for b,w,s in p['bundles']),345)

    def test_seed_registry(self):
        audit=runner.read(OUTPUT/'seed_audit.json')
        self.assertEqual(audit['collisions'],[])
        self.assertGreater(len(audit['scanned']),20)

    def test_common_price_and_pool_parity(self):
        from utils.sf_hlm_action_ranker import public_payload
        samples=sorted((OUTPUT/'smoke').glob('*/G/samples/0000/public.json'))
        self.assertTrue(samples)
        rows=runner.read(samples[0])['rows']
        for row in rows:row['options']={int(k):v for k,v in row['options'].items()}
        from utils.sf_hlm_priced_runtime import PricedPolicy
        with FAST.FastRuntime() as rt:
            native=PricedPolicy(rt,None,PRICES,PRICES['powers'])
            for component,original in [('B0','binary_price'),('Mlambda','mixture_price')]:
                policy,kind=runner.make_policy(rt,component,PRICES)
                left=policy.run(rows,num_uavs=5,kind=kind,trace=True)
                right=native.run(rows,num_uavs=5,kind=original,trace=True)
                self.assertEqual(runner.normal(left),runner.normal(right))
            self.assertEqual(PRICES['prices']['binary']['price'],0.)

    def test_fixed_branch_probability_intervention(self):
        from utils.sf_hlm_cache_mixture import probability
        part=dict(base=.5,work=.1,p=.9,hit_base=.2,miss_base=.8,deterministic=False,fallback=None)
        boundary=part['hit_base']+part['work']*2+.01
        self.assertEqual(probability(part,2,boundary,.01),.9)
        changed=copy.deepcopy(part);changed['p']=.5
        self.assertEqual(probability(changed,2,boundary,.01),.5)
        self.assertEqual((changed['hit_base'],changed['miss_base']),(.2,.8))
        self.assertEqual(probability(part,2,1.01,.01),1.)
        self.assertEqual(probability(part,2,.40,.01),0.)
        for value in [0.,1.,1.-1e-14]:
            changed['p']=value
            self.assertEqual(probability(changed,2,boundary,.01),value)
        changed['fallback']='invalid_decomposition'
        self.assertEqual(probability(changed,2,.71,.01),1.)

    def test_block_rate_counting(self):
        # Unequal denominators: averaging episode rates would give .55.
        successes=[1,9];generated=[1,99];admitted=[1,49]
        dsr=sum(successes)/sum(generated)
        self.assertEqual(dsr,.1)
        self.assertAlmostEqual(dsr,(sum(admitted)/sum(generated))*(sum(successes)/sum(admitted)))
        self.assertEqual(120./100.,1.2)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--archive',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();OUTPUT=args.output
    _,RUNTIME,FAST,CORE,PRICES=runner.boot(args,620001)
    unittest.main(argv=[sys.argv[0]])
