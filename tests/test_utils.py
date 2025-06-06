# File: tests/test_utils.py

import unittest
from adaptive_tsp.utils import compute_distance_matrix, two_opt

class TestUtils(unittest.TestCase):
    def test_compute_distance_matrix(self):
        points = [(0, 0), (3, 0), (0, 4)]
        mat = compute_distance_matrix(points)

        self.assertAlmostEqual(mat[0][1], 3.0)
        self.assertAlmostEqual(mat[0][2], 4.0)
        self.assertAlmostEqual(mat[1][2], 5.0)

        self.assertEqual(mat[1][0], mat[0][1])
        self.assertEqual(mat[2][0], mat[0][2])
        self.assertEqual(mat[2][1], mat[1][2])

    def test_two_opt_improves(self):
        points = [(0,0), (0,1), (1,1), (1,0)]
        from adaptive_tsp.utils import tour_length
        matrix = compute_distance_matrix(points)
        init_tour = [0, 2, 1, 3]
        init_len = tour_length(init_tour, matrix)
        self.assertAlmostEqual(init_len, 4 * math.hypot(1, 1))
        improved = two_opt(init_tour, matrix)
        final_len = tour_length(improved, matrix)
        self.assertAlmostEqual(final_len, 4.0, places=5)

if __name__ == '__main__':
    unittest.main()
