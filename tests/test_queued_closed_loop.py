"""Pure protocol/statistical/vectorized-actor tests; no local rollouts or Torch."""
import unittest
import numpy as np
from utils.queued_closed_loop import (design, validate_job, paired_summary, select_method,
    relative_scores, pack_rows, packed_statistics, batch_improve)
from marl_models.queued_joint_mpo import policy,loss_gradient
from utils.queued_joint_mpo import improve

class ClosedLoopTests(unittest.TestCase):
    def test_disjoint_reserved_workloads_and_training_seeds(self):
        p=design('pilot');f=design('formal')
        self.assertEqual(len(p['runs']),6);self.assertEqual(len(f['runs']),15)
        self.assertEqual(len(p['evaluation_workloads']),8)
        a={w for r in p['runs'] for ws in r['training'] for w in ws}
        b={w for r in f['runs'] for ws in r['training'] for w in ws}
        self.assertEqual(len(a),72);self.assertEqual(len(b),1440)
        sets=[a,b,set(p['evaluation_workloads']),set(f['evaluation_workloads'])]
        for i,x in enumerate(sets):
            for y in sets[i+1:]:self.assertFalse(x&y)
        self.assertFalse({r['seed'] for r in p['runs']}&{r['seed'] for r in f['runs']})
    def test_pilot_cannot_touch_formal_test(self):
        p=design('pilot');f=design('formal')
        with self.assertRaises(ValueError):validate_job(p,'evaluation',0,f['evaluation_workloads'][0])
        validate_job(p,'evaluation',0,p['evaluation_workloads'][0])
        with self.assertRaises(ValueError):validate_job(p,'training',0,p['runs'][1]['training'][0][0])
    def test_clustered_summary_constant_effect(self):
        g=np.full((3,2,4,2),100.);a=g*.4;b=g*.3
        z=paired_summary(a,b,g,draws=80)
        np.testing.assert_allclose(z['ci95'],[.1,.1]);self.assertEqual(z['training_repeats_per_fixed_parent'],2)
        self.assertEqual(z['workloads'],4);self.assertEqual(z['action_replicas'],2)
    def test_pairing_rejects_nonconserved_counts(self):
        with self.assertRaises(ValueError):paired_summary(np.array([[[[3.]]]]),np.array([[[[0.]]]]),np.array([[[[2.]]]]),draws=10)
    def summaries(self):
        methods=['relative_joint','short_joint','relative_decomposed','short_decomposed','shuffled_joint']
        c={}
        for m in methods:
            for other in ['initial']+methods:
                if m!=other:c[m+'--'+other]=dict(mean=.005,ci95=[.001,.009],by_parent=[.005]*3)
        return c
    def test_negative_pilot_freezes_no_go(self):
        c=self.summaries()
        for v in c.values():v.update(mean=-.001,ci95=[-.003,.001])
        self.assertEqual(select_method(c,True)['status'],'NO_GO')
    def test_prefers_short_when_correct_labels_not_distinguished(self):
        c=self.summaries();c['relative_joint--shuffled_joint']['ci95']=[-.01,.01]
        self.assertEqual(select_method(c,True)['selected_method'],'short_joint')
    def test_relative_selected_when_both_specific_controls_pass(self):
        self.assertEqual(select_method(self.summaries(),True)['selected_method'],'relative_joint')
        self.assertEqual(select_method(self.summaries(),False)['status'],'NO_GO')
    def test_scores_root_centering_and_known_empty_boundary(self):
        model=dict(mean=[0.],scale=[1.],models={'root_centered':{'beta':[2.]},'root_shuffled':{'beta':[-1.]}})
        p=np.array([[2.,1.]]);x=np.array([[[1.],[3.]]])
        np.testing.assert_allclose(relative_scores(model,x,p,np.zeros((1,2),bool),'root_centered'),[[-1.5,1.5]])
        np.testing.assert_allclose(relative_scores(model,x,p,np.ones((1,2),bool),'root_centered'),[[.5,-.5]])
    def rows(self):
        rng=np.random.default_rng(11);rows=[]
        for k,m in [(2,3),(1,2),(3,1)]:
            rows.append(dict(a=rng.normal(size=(k,4)),m=rng.normal(size=(k,m,3)),prior=np.ones(k)/k,q=rng.normal(size=(k,m))))
        return rows
    def test_packed_statistics_matches_ragged_scalar_actor(self):
        rows=self.rows();theta=np.arange(7)*.03;old=np.zeros(7)
        for mode in ['joint','decomposed']:
            packed=pack_rows(rows,old,mode);target,info=batch_improve(packed['old'],packed['q'],.1);packed['target']=target
            z=packed_statistics(packed,theta,old)
            loss=0.;gradient=np.zeros(7)
            for i,r in enumerate(rows):
                k,m=r['m'].shape[:2];l,g=loss_gradient(r,theta,target[i,:k,:m],mode);loss+=l;gradient+=g
            np.testing.assert_allclose(z['gradient'],gradient,atol=1e-12)
            self.assertAlmostEqual(z['loss'],loss);self.assertAlmostEqual(z['kl'],z['akl']+z['mkl'])
    def test_batch_estep_matches_scalar_global_dual(self):
        rows=self.rows();packed=pack_rows(rows,np.zeros(7),'joint')
        a,info=batch_improve(packed['old'],packed['q'],.1)
        b,ref=improve([policy(r,np.zeros(7)) for r in rows],[r['q'] for r in rows],.1)
        for i,(r,x) in enumerate(zip(rows,b)):
            k,m=r['m'].shape[:2];np.testing.assert_allclose(a[i,:k,:m],x,atol=1e-10)
        self.assertAlmostEqual(info['kl'],ref['kl']);self.assertGreaterEqual(info['gain'],-1e-10)

if __name__=='__main__':unittest.main()
