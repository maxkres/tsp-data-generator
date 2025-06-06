# File: benchmarks/run_benchmarks.py

import os
import time
import csv
import numpy as np

from data_generator.api import generate_dataset
from adaptive_tsp.adaptive_tsp import AdaptiveTSP
from adaptive_tsp.utils import tour_length
import networkx as nx

OUTPUT_DIR = 'benchmarks/results'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def solve_with_quickml(points):
    G = nx.complete_graph(len(points))
    for u in range(len(points)):
        for v in range(u+1, len(points)):
            dist = np.hypot(points[u][0]-points[v][0], points[u][1]-points[v][1])
            G[u][v]['weight'] = dist
            G[v][u]['weight'] = dist
    start = time.time()
    tour = nx.approximation.traveling_salesman_problem(G, weight='weight')
    elapsed = time.time() - start
    if tour[-1] != tour[0]:
        tour.append(tour[0])
    return tour[:-1], elapsed

def run_single(n, distribution):
    if distribution == 'uniform':
        points, _ = generate_dataset(mode='uniform', n=n, x_min=0, x_max=1, y_min=0, y_max=1, seed=42)
    elif distribution == 'cluster':
        points, _ = generate_dataset(mode='cluster', n=n, x_min=0, x_max=1, y_min=0, y_max=1,
                                     clusters=5, cluster_std=0.05, seed=42)
    else:
        side = int(math.sqrt(n))
        xs = np.linspace(0, 1, side)
        ys = np.linspace(0, 1, side)
        points = [(float(xs[i]), float(ys[j])) for i in range(side) for j in range(side)]
        points = points[:n]

    # 1) AdaptiveTSP
    solver = AdaptiveTSP(points, iterations=2000, refine_rounds=2, som_lr_initial=0.8, som_lr_final=0.01,
                         som_radius_final=1.0, max_mapping_dist=0.2)
    start = time.time()
    tour_adaptive = solver.solve()
    time_adaptive = time.time() - start
    length_adaptive = tour_length(tour_adaptive, solver.matrix)

    # 2) NetworkX
    tour_nx, time_nx = solve_with_quickml(points)
    length_nx = tour_length(tour_nx, solver.matrix)

    # 3) Static SOM (AdaptiveGrid без add/remove, refine_rounds=1)
    static_solver = AdaptiveTSP(points, iterations=2000, som_initial_nodes=n,
                                som_lr_initial=0.8, som_lr_final=0.01, som_radius_final=1.0,
                                refine_rounds=1, max_mapping_dist=None)
    start = time.time()
    tour_static = static_solver.solve()
    time_static = time.time() - start
    length_static = tour_length(tour_static, solver.matrix)

    return {
        'n': n,
        'distribution': distribution,
        'adaptive_len': length_adaptive,
        'adaptive_time': time_adaptive,
        'nx_len': length_nx,
        'nx_time': time_nx,
        'static_som_len': length_static,
        'static_som_time': time_static
    }

if __name__ == "__main__":
    ns = [50, 100, 200]
    distributions = ['uniform', 'cluster', 'grid']

    results_file = os.path.join(OUTPUT_DIR, 'benchmark_results.csv')
    with open(results_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            'n', 'distribution',
            'adaptive_len', 'adaptive_time',
            'nx_len', 'nx_time',
            'static_som_len', 'static_som_time'
        ])
        for n in ns:
            for dist in distributions:
                print(f"Benchmark: n={n}, dist={dist}")
                res = run_single(n, dist)
                writer.writerow([
                    res['n'], res['distribution'],
                    f"{res['adaptive_len']:.6f}", f"{res['adaptive_time']:.4f}",
                    f"{res['nx_len']:.6f}", f"{res['nx_time']:.4f}",
                    f"{res['static_som_len']:.6f}", f"{res['static_som_time']:.4f}"
                ])
    print(f"Бенчмарки завершены, результаты в {results_file}")
