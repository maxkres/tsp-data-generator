# File: tests/test_adaptive_tsp.py

import unittest
from adaptive_tsp.adaptive_tsp import AdaptiveTSP

class TestAdaptiveTSP(unittest.TestCase):
    def test_tour_covers_all(self):
        points = [(0,0), (1,0), (0,1), (-1,0), (0,-1)]
        solver = AdaptiveTSP(
            points=points,
            iterations=100,
            som_lr_initial=0.5,
            som_lr_final=0.01,
            som_radius_final=1.0,
            refine_rounds=1,
            max_mapping_dist=None
        )
        tour = solver.solve()
        self.assertEqual(len(tour), 5)
        self.assertEqual(set(tour), set(range(5)))
        init_len, final_len = solver.get_lengths()
        self.assertGreater(init_len, 0)
        self.assertGreater(final_len, 0)

if __name__ == '__main__':
    unittest.main()
