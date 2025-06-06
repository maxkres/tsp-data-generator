# File: data_generator/src/tsp_generator/output.py

import math
import csv
import json
from typing import List, Tuple
import numpy as np

def compute_distance_matrix(points: List[Tuple[float, float]]) -> List[List[float]]:
    n = len(points)
    mat = [[0.0] * n for _ in range(n)]
    for i, (xi, yi) in enumerate(points):
        for j in range(i + 1, n):
            xj, yj = points[j]
            d = math.hypot(xi - xj, yi - yj)
            mat[i][j] = mat[j][i] = d
    return mat

def save_csv(path: str, points: List[Tuple[float, float]]):
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['index', 'x', 'y'])
        for idx, (x, y) in enumerate(points, 1):
            writer.writerow([idx, f"{x:.6f}", f"{y:.6f}"])

def save_tsp(path: str, points: List[Tuple[float, float]]):
    n = len(points)
    with open(path, 'w') as f:
        f.write("NAME: generated\n")
        f.write("TYPE: TSP\n")
        f.write(f"DIMENSION: {n}\n")
        f.write("EDGE_WEIGHT_TYPE: EUC_2D\n")
        f.write("NODE_COORD_SECTION\n")
        for idx, (x, y) in enumerate(points, 1):
            f.write(f"{idx} {x:.6f} {y:.6f}\n")
        f.write("EOF\n")

def save_json(path: str, points: List[Tuple[float, float]], matrix: List[List[float]]):
    data = {
        "points": [{"x": x, "y": y} for x, y in points],
        "distance_matrix": matrix
    }
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)

def save_mat(path: str, matrix: List[List[float]]):
    arr = np.array(matrix)
    if path.lower().endswith('.npy'):
        np.save(path, arr)
    else:
        np.savetxt(path, arr, delimiter=',', fmt="%.6f")
