import unittest


class FrameworkAdapterContract(unittest.TestCase):
    def test_random_cap_and_exogenous_search_seed_are_matched(self):
        from scripts.analysis.sf_hlm_framework_evaluation import spec
        left=spec('R10',1520001,1560001,0,4);right=spec('R16',1520001,1560001,0,4)
        self.assertEqual(left['seed'],right['seed']);self.assertEqual((left['kind'],right['kind']),('random','random'));self.assertEqual((left['cap'],right['cap']),(10,16))
        self.assertNotEqual(left['seed'],spec('R10',1520001,1560001,1,4)['seed'])

    def test_primary_and_hgb_use_frozen_existing_kinds(self):
        from scripts.analysis.sf_hlm_framework_evaluation import spec,METHODS
        self.assertEqual(len(METHODS),9);self.assertEqual(spec('G',1,2,0,0)['kind'],'native_G');self.assertEqual(spec('mixture_price',1,2,0,0)['kind'],'mixture_price');self.assertEqual(spec('HGB_probability_price',1,2,0,0)['kind'],'probability_price')
        with self.assertRaises(ValueError):spec('L16',1,2,0,0)

    def test_no_quota_actor_is_silently_used(self):
        from scripts.analysis.sf_hlm_framework_evaluation import ForbiddenQuotaActor
        with self.assertRaises(AssertionError):ForbiddenQuotaActor().act(None)

    def test_new_budget_registration_is_limited_to_random_ten(self):
        from utils.sf_hlm_fast_runtime import FastRuntime
        rt=FastRuntime()
        for cap,kind in ((10,'learned'),(10,'heuristic'),(12,'random')):
            with self.assertRaises(ValueError):rt.truncated_search([],num_uavs=5,kind=kind,cap=cap)

    def test_comparison_alias_preserves_primary_price_without_mutation(self):
        from scripts.analysis.sf_hlm_framework_evaluation import comparison_prices
        prices=dict(prices=dict(mixture=dict(price=.002,normalizer=3.),binary=dict(price=0.,normalizer=2.)),powers=dict(backhaul_rx=.1));aliased=comparison_prices(prices)
        self.assertNotIn('prior',prices['prices']);self.assertIs(aliased['prices']['prior'],prices['prices']['mixture']);self.assertEqual(aliased['prices']['mixture'],prices['prices']['mixture']);self.assertEqual(aliased['powers'],prices['powers'])

    def test_full_runner_preserves_preexisting_shuffled_tasks(self):
        from scripts.analysis.sf_hlm_framework_confirmation import design
        from scripts.analysis.sf_hlm_framework_evaluation import runner_design
        frozen=dict(smoke=False,**design());actual=runner_design(frozen,False)
        self.assertEqual(actual['tasks'],frozen['planned_evaluation_tasks']);self.assertEqual(len(actual['tasks']),2700);self.assertEqual(len(actual['physical_replay_tasks']),180)
        self.assertEqual(actual['block_order'],list(dict.fromkeys(t[1] for t in actual['tasks'])))
        for block in actual['block_order']:
            self.assertEqual(len([t for t in actual['tasks'] if t[1]==block]),135)
        with self.assertRaises(ValueError):runner_design(frozen,True)

    def test_smoke_never_uses_fresh_confirmation_workloads(self):
        from scripts.analysis.sf_hlm_framework_confirmation import design,SMOKE_LOAD
        from scripts.analysis.sf_hlm_framework_evaluation import runner_design
        p=runner_design(dict(smoke=True,**design(True)),True)
        self.assertEqual(p['workloads'],[SMOKE_LOAD]);self.assertEqual(len(p['tasks']),18);self.assertEqual(p['steps'],16);self.assertEqual(p['physical_replay_tasks'],p['tasks']);self.assertEqual({t[3] for t in p['tasks']},{0,1})


if __name__=='__main__':unittest.main()
