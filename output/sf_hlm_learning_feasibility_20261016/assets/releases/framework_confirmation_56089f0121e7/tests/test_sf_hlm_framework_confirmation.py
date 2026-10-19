import unittest


class FreshFrameworkContract(unittest.TestCase):
    def test_new_units_are_disjoint_and_have_fixed_last_checkpoint_recipe(self):
        from scripts.analysis.sf_hlm_framework_confirmation import design,BLOCKS,TRAIN_LOADS,EVAL_LOADS,SMOKE_BLOCK,SMOKE_LOAD
        d=design();self.assertEqual(len(BLOCKS),20);self.assertEqual(len(set(BLOCKS)|set(TRAIN_LOADS)|set(EVAL_LOADS)),45)
        self.assertFalse(set(BLOCKS)&set(range(1060001,1060011)));self.assertFalse(set(BLOCKS)&set(range(620001,620006)))
        self.assertNotIn(SMOKE_BLOCK,BLOCKS);self.assertNotIn(SMOKE_LOAD,TRAIN_LOADS);self.assertEqual(d['foundation_episodes'],200);self.assertEqual(d['steps'],1000)

    def test_all_prespecified_methods_and_episodes_retained(self):
        from scripts.analysis.sf_hlm_framework_confirmation import design,METHODS
        d=design();tasks=d['planned_evaluation_tasks'];self.assertEqual(len(tasks),2700);self.assertEqual(len({tuple(t) for t in tasks}),2700)
        for method in METHODS:self.assertEqual(sum(t[0]==method for t in tasks),300)
        self.assertEqual(d['primary_comparisons'],[['mixture_price','G'],['mixture_price','binary_price']]);self.assertFalse(d['unique_learning_increment_resolved'])

    def test_smoke_never_uses_formal_training_units(self):
        from scripts.analysis.sf_hlm_framework_confirmation import design
        formal=design();smoke=design(True);self.assertEqual(len(smoke['tasks']),1);self.assertFalse(set(smoke['confirmation_blocks'])&set(formal['confirmation_blocks']))
        self.assertEqual(smoke['foundation_episodes'],2);self.assertEqual(smoke['steps'],8);self.assertEqual(formal['gates']['positive_blocks'],16)


if __name__=='__main__':unittest.main()
