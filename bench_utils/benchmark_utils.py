import os
import sys
import time
import math
import random

import numpy as np
import pandas as pd

_this_dir = os.path.dirname(__file__)
_project_root = os.path.abspath(os.path.join(_this_dir, '..'))
sys.path.insert(0, _project_root)

from adaptive_tsp.adaptive_tsp import AdaptiveTSP
from adaptive_tsp.utils import two_opt, tour_length

_data_gen_src = os.path.join(_project_root, 'data_generator', 'src')
sys.path.insert(0, _data_gen_src)
from tsp_generator.generators.uniform import generate_uniform
from tsp_generator.generators.circle import generate_circle
from tsp_generator.generators.cluster import generate_cluster
from tsp_generator.generators.osm import generate_osm


def generate_synthetic(distribution: str, n: int, **kwargs):
    if distribution == 'uniform':
        x_min = kwargs.get('x_min', 0.0)
        x_max = kwargs.get('x_max', 1.0)
        y_min = kwargs.get('y_min', 0.0)
        y_max = kwargs.get('y_max', 1.0)
        seed = kwargs.get('seed', 0)
        pts = generate_uniform(n, x_min, x_max, y_min, y_max, seed)
        return pts

    elif distribution == 'ring':
        radius = kwargs.get('radius', 0.5)
        on_boundary = kwargs.get('on_boundary', True)
        seed = kwargs.get('seed', 0)
        pts = generate_circle(n, radius, on_boundary, seed)
        return pts

    elif distribution == 'clusters':
        x_min = kwargs.get('x_min', 0.0)
        x_max = kwargs.get('x_max', 1.0)
        y_min = kwargs.get('y_min', 0.0)
        y_max = kwargs.get('y_max', 1.0)
        clusters = kwargs.get('clusters', 3)
        std = kwargs.get('std', 0.05)
        seed = kwargs.get('seed', 0)
        pts = generate_cluster(n, x_min, x_max, y_min, y_max, clusters, std, seed)
        return pts

    else:
        raise ValueError(f"Неизвестный distribution='{distribution}'")


def load_osm(place_name: str, num_cities: int, poi_key: str = 'amenity', poi_value: str = 'restaurant'):
    pts = generate_osm(place_name, num_cities, poi_key, poi_value)
    return pts


def load_tsplib(tsplib_filepath: str):
    points = []
    with open(tsplib_filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    start = None
    for i, line in enumerate(lines):
        if line.strip().upper().startswith('NODE_COORD_SECTION'):
            start = i + 1
            break
    if start is None:
        raise RuntimeError(f"В файле {tsplib_filepath} нет 'NODE_COORD_SECTION'")

    for line in lines[start:]:
        parts = line.strip().split()
        if len(parts) < 3:
            break
        try:
            _, x_str, y_str = parts[:3]
            x = float(x_str)
            y = float(y_str)
            points.append((x, y))
        except:
            break

    return points


def compute_baseline_nn_2opt(points):
    start_time = time.time()
    n = len(points)
    mat = [[0.0]*n for _ in range(n)]
    for i in range(n):
        xi, yi = points[i]
        for j in range(i+1, n):
            xj, yj = points[j]
            d = math.hypot(xi - xj, yi - yj)
            mat[i][j] = mat[j][i] = d

    visited = [False]*n
    tour = [0]
    visited[0] = True
    for _ in range(n-1):
        last = tour[-1]
        nearest = None
        best_d = float('inf')
        for j in range(n):
            if not visited[j] and mat[last][j] < best_d:
                best_d = mat[last][j]
                nearest = j
        tour.append(nearest)
        visited[nearest] = True

    length_initial = tour_length(tour, mat)
    improved = two_opt(tour, mat)
    length_final = tour_length(improved, mat)
    total_time = time.time() - start_time
    return improved, length_initial, length_final, total_time


def evaluate_tour_length(points, tour):
    n = len(points)
    mat = [[0.0]*n for _ in range(n)]
    for i in range(n):
        xi, yi = points[i]
        for j in range(i+1, n):
            xj, yj = points[j]
            d = math.hypot(xi - xj, yi - yj)
            mat[i][j] = mat[j][i] = d
    return tour_length(tour, mat)


def read_optimum_tsplib(tsplib_filepath: str):
    base, _ = os.path.splitext(tsplib_filepath)
    candidates = [base + '.opt', base + '.opt.tour', base + '.opt.txt']
    for c in candidates:
        if os.path.exists(c):
            try:
                with open(c, 'r', encoding='utf-8') as f:
                    first_line = f.readline().strip()
                    val = float(first_line)
                    return val
            except:
                continue
    return None
