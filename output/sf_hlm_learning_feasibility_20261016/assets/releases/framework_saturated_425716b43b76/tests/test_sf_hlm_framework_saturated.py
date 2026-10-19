import collections
import unittest


class SaturatedFrameworkContract(unittest.TestCase):
    def test_frozen_tasks_and_twenty_block_inference_are_unchanged(self):
        from scripts.analysis.sf_hlm_framework_confirmation import design as foundation_design
        from scripts.analysis.sf_hlm_framework_saturated import design
        fp=dict(smoke=False,**foundation_design());p=design(fp,False)
        self.assertEqual(p['tasks'],fp['planned_evaluation_tasks']);self.assertEqual(p['gates'],fp['gates']);self.assertEqual(p['primary_comparisons'],fp['primary_comparisons']);self.assertEqual(len(p['blocks']),20);self.assertEqual(len(p['tasks']),2700)

    def test_three_hundred_paired_bundles_preserve_every_method_and_order(self):
        from scripts.analysis.sf_hlm_framework_confirmation import design as foundation_design
        from scripts.analysis.sf_hlm_framework_saturated import design,bundle_tasks,METHODS
        p=design(dict(smoke=False,**foundation_design()),False);self.assertEqual(len(p['bundle_order']),300);actual=[]
        for bundle in p['bundle_order']:
            rows=bundle_tasks(p,bundle);self.assertEqual(rows,[t for t in p['tasks'] if t[1:4]==bundle]);self.assertEqual(len(rows),9);self.assertEqual({t[0] for t in rows},set(METHODS));actual+=rows
        self.assertEqual(collections.Counter(map(tuple,actual)),collections.Counter(map(tuple,p['tasks'])))

    def test_capacity_and_fixed_wave_sizes_do_not_add_samples(self):
        from scripts.analysis.sf_hlm_framework_confirmation import design as foundation_design
        from scripts.analysis.sf_hlm_framework_saturated import design
        p=design(dict(smoke=False,**foundation_design()),False);self.assertEqual(p['workers'],120);self.assertEqual(p['wave_sizes'],[120,120,60]);self.assertEqual(p['audit_workers'],120);self.assertEqual(len(p['physical_replay_tasks']),180)

    def test_smoke_uses_only_its_original_independent_load(self):
        from scripts.analysis.sf_hlm_framework_confirmation import design as foundation_design,SMOKE_LOAD
        from scripts.analysis.sf_hlm_framework_saturated import design
        p=design(dict(smoke=True,**foundation_design(True)),True);self.assertEqual(p['workloads'],[SMOKE_LOAD]);self.assertEqual(p['workers'],2);self.assertEqual(p['wave_sizes'],[2]);self.assertEqual(p['steps'],16);self.assertEqual(len(p['tasks']),18);self.assertEqual(p['physical_replay_tasks'],p['tasks'])

    def test_missing_or_duplicate_method_in_bundle_is_rejected(self):
        from scripts.analysis.sf_hlm_framework_confirmation import design as foundation_design
        from scripts.analysis.sf_hlm_framework_saturated import design,bundle_tasks
        p=design(dict(smoke=True,**foundation_design(True)),True);bundle=p['bundle_order'][0];rows=bundle_tasks(p,bundle);p['tasks'].remove(rows[0])
        with self.assertRaises(ValueError):bundle_tasks(p,bundle)
        p['tasks'].append(rows[1])
        with self.assertRaises(ValueError):bundle_tasks(p,bundle)


if __name__=='__main__':unittest.main()
