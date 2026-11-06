import copy
import unittest
import numpy as np
from utils.queued_root_advantage import fit_models,predict,tie_weights,pairwise_loss,bootstrap_blocks

class RootAdvantageTests(unittest.TestCase):
    def data(self):
        rng=np.random.default_rng(10);rows=[]
        for k in range(12):
            x=rng.normal(size=(4,5));p=rng.normal(size=4)
            y=p+x@np.array([2.,-1.,.5,0.,1.])+100*k
            rows.append(dict(x=x,p=p,y=y,time=75 if k%2 else 175,key=str(k)))
        return rows
    def test_root_label_shifts_do_not_change_root_fit(self):
        rows=self.data();other=copy.deepcopy(rows)
        for i,r in enumerate(other):r['y']+=10000*(i+1)
        a=fit_models(rows);b=fit_models(other)
        np.testing.assert_allclose(a['models']['root_centered']['beta'],b['models']['root_centered']['beta'],atol=1e-9)
    def test_pairwise_and_root_centered_loss_are_equal(self):
        r=np.array([3.,-4.,7.,2.]);self.assertAlmostEqual(pairwise_loss(r),np.mean((r-r.mean())**2))
    def test_training_only_scaling_and_positive_ranking(self):
        rows=self.data();model=fit_models(rows,alpha=1e-8);before=copy.deepcopy(model)
        z=predict(model,'root_centered',rows[0]['x'],rows[0]['p'],rows[0]['time'])
        self.assertLess(abs(z.mean()),1e-10)
        self.assertGreater(np.corrcoef(z,rows[0]['y'])[0,1],.99)
        predict(model,'root_centered',np.full((4,5),1e8),np.arange(4.),75)
        self.assertEqual(model,before)
    def test_uniform_tie_weights_and_validation(self):
        np.testing.assert_array_equal(tie_weights([3.,3.,0.,3.]),[1/3,1/3,0.,1/3])
        with self.assertRaises(ValueError):tie_weights([np.nan,1])
        with self.assertRaises(ValueError):fit_models([])
    def test_candidate_permutation_does_not_change_primary_model(self):
        rows=self.data();other=copy.deepcopy(rows)
        for r in other:
            for key in ('x','p','y'):r[key]=r[key][[2,0,3,1]]
        a=fit_models(rows);b=fit_models(other)
        np.testing.assert_allclose(a['models']['root_centered']['beta'],b['models']['root_centered']['beta'],atol=1e-10)
    def test_variable_bank_sizes_keep_equal_root_weight(self):
        rows=[dict(x=[[0.]],p=[0.],y=[3.],time=75,key='one'),
              dict(x=[[10.],[10.],[10.],[10.]],p=[0.]*4,y=[8.]*4,time=175,key='four')]
        fitted=fit_models(rows)
        self.assertAlmostEqual(fitted['mean'][0],5.)
        np.testing.assert_allclose(predict(fitted,'root_centered',[[0.]],[0.],75),[0.])

    def test_workload_bootstrap_preserves_shared_shift_cancellation(self):
        x=np.array([[2.,2.,2.],[2.,2.,2.]])
        result=bootstrap_blocks(x,draws=100)
        np.testing.assert_allclose(result['ci95'],[2.,2.]);np.testing.assert_allclose(result['simultaneous98_333'],[2.,2.])

if __name__=='__main__':unittest.main()
