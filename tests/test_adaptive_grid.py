# File: tests/test_adaptive_grid.py

import unittest
from adaptive_tsp.adaptive_grid import AdaptiveGrid

class TestAdaptiveGrid(unittest.TestCase):
    def test_dynamic_add_and_remove(self):
        points = [(0, 0), (1, 0), (0, 1)]
        grid = AdaptiveGrid(
            points=points,
            initial_nodes=3,
            iterations=50,
            lr_initial=0.5,
            lr_final=0.01,
            radius_initial=1.0,
            radius_final=0.5
        )
        initial_m = grid.m
        grid.train()
        self.assertGreaterEqual(grid.m, 3)
        nodes = grid.get_nodes()
        for (x, y) in nodes:
            self.assertTrue(-1 <= x <= 2)
            self.assertTrue(-1 <= y <= 2)

if __name__ == '__main__':
    unittest.main()
