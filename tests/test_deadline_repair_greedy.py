import unittest
from utils.deadline_repair_online import greedy_assignment

class TestGreedy(unittest.TestCase):
    def test_empty_and_prefix_load(self):
        self.assertEqual(greedy_assignment([]), ((),0))
        rows=[{'deadline':1.,'options':{0:{'target':0,'base':0.,'work':.6,'energy':1.,'compute_energy':0.},1:{'target':'mbs','base':1.,'work':0.,'energy':2.,'compute_energy':0.}}} for _ in range(2)]
        self.assertEqual(greedy_assignment(rows),((0,1),4))
    def test_mask_ties_and_determinism(self):
        option={'target':0,'base':1.,'work':0.,'energy':2.,'compute_energy':0.}
        rows=[{'deadline':2.,'options':{3:dict(option),2:dict(option)}}]*30
        self.assertEqual(greedy_assignment(rows),((2,)*30,60))
        self.assertEqual(greedy_assignment(rows),greedy_assignment(rows))
        with self.assertRaises(ValueError):greedy_assignment([{'options':{}}])
if __name__=='__main__':unittest.main()
