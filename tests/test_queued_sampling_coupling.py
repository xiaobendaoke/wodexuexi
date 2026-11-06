import unittest
from utils.queued_sampling_coupling import gumbel_choice,semantic_labels


class CouplingTest(unittest.TestCase):
    def test_permutation_preserves_semantic_choice(self):
        p=[.1,.3,.6];labels=['a','b','c']
        for i in range(100):
            a=gumbel_choice(p,labels,i);b=gumbel_choice(p[::-1],labels[::-1],i)
            self.assertEqual(labels[a],labels[::-1][b])

    def test_categorical_frequencies_and_zero_mask(self):
        counts=[0]*4
        for i in range(10000):counts[gumbel_choice([.1,.3,.6,0],['a','b','c','masked'],i)]+=1
        for actual,want in zip(counts,[.1,.3,.6,0]):self.assertAlmostEqual(actual/10000,want,delta=.02)

    def test_all_candidates_have_unique_semantics(self):
        from tests.test_queued_service_learning import observation
        from utils.queued_long_value import LongAdapter
        obs=observation();adapter=LongAdapter(True);adapter.prepare(obs)
        labels=semantic_labels(obs,adapter.history.weights,adapter.plans)
        self.assertEqual(len(labels),len(set(labels)))
