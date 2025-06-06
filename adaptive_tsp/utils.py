import math
from typing import List, Tuple
from scipy.spatial import KDTree

def euclidean_distance(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])

def compute_distance_matrix(points: List[Tuple[float, float]]) -> List[List[float]]:
    n = len(points)
    mat = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            d = euclidean_distance(points[i], points[j])
            mat[i][j] = mat[j][i] = d
    return mat

def tour_length(tour: List[int], matrix: List[List[float]]) -> float:
    n = len(tour)
    length = 0.0
    for i in range(n - 1):
        length += matrix[tour[i]][tour[i + 1]]
    length += matrix[tour[-1]][tour[0]]
    return length

def two_opt_swap(tour: List[int], i: int, k: int) -> List[int]:
    return tour[0:i] + tour[i:k + 1][::-1] + tour[k + 1:]

def two_opt(tour: List[int], matrix: List[List[float]]) -> List[int]:
    improved = True
    best_tour = tour
    best_length = tour_length(tour, matrix)
    n = len(tour)

    while improved:
        improved = False
        for i in range(1, n - 1):
            for k in range(i + 1, n):
                new_tour = two_opt_swap(best_tour, i, k)
                new_length = tour_length(new_tour, matrix)
                if new_length < best_length:
                    best_tour = new_tour
                    best_length = new_length
                    improved = True
                    break
            if improved:
                break

    return best_tour

def build_kdtree(points: List[Tuple[float, float]]) -> KDTree:
    coords = [tuple(pt) for pt in points]
    return KDTree(coords)
