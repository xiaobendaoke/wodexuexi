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


if __name__=='__main__':unittest.main()
