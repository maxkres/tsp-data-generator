# File: data_generator/src/tsp_generator/cli.py

import os
import sys
import argparse

from typing import List, Tuple
from tsp_generator.generators.uniform import generate_uniform
from tsp_generator.generators.cluster import generate_cluster
from tsp_generator.generators.circle import generate_circle
from tsp_generator.generators.osm import generate_osm

from tsp_generator.output import (
    save_csv, save_tsp, compute_distance_matrix,
    save_json, save_mat
)
from tsp_generator.visualize import visualize

def main():
    parser = argparse.ArgumentParser(description="Генератор данных для евклидового TSP")
    parser.add_argument('--mode', choices=['uniform', 'cluster', 'circle', 'osm'], default='uniform')
    parser.add_argument('-n', '--num-cities', type=int, default=100)
    parser.add_argument('--x-min', type=float, default=0.0)
    parser.add_argument('--x-max', type=float, default=1.0)
    parser.add_argument('--y-min', type=float, default=0.0)
    parser.add_argument('--y-max', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--clusters', type=int, default=5)
    parser.add_argument('--cluster-std', type=float, default=0.05)
    parser.add_argument('--radius', type=float, default=1.0)
    parser.add_argument('--on-boundary', action='store_true')
    parser.add_argument('--place', type=str, default=None)
    parser.add_argument('--poi-key', type=str, default='amenity')
    parser.add_argument('--poi-value', type=str, default='restaurant')
    parser.add_argument('--output-csv', type=str)
    parser.add_argument('--output-tsp', type=str)
    parser.add_argument('--output-json', type=str)
    parser.add_argument('--output-mat', type=str)
    parser.add_argument('--output-png', type=str)
    parser.add_argument('--connect', action='store_true')
    parser.add_argument('--output-dir', type=str, default='output')

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    points: List[Tuple[float, float]] = []

    if args.mode == 'uniform':
        points = generate_uniform(
            args.num_cities,
            args.x_min, args.x_max,
            args.y_min, args.y_max,
            args.seed
        )
    elif args.mode == 'cluster':
        points = generate_cluster(
            args.num_cities,
            args.x_min, args.x_max,
            args.y_min, args.y_max,
            args.clusters, args.cluster_std,
            args.seed
        )
    elif args.mode == 'circle':
        points = generate_circle(
            args.num_cities,
            args.radius,
            args.on_boundary,
            args.seed
        )
    elif args.mode == 'osm':
        if not args.place:
            print("Ошибка: для --mode osm нужно указать --place")
            sys.exit(1)
        points = generate_osm(
            place_name=args.place,
            num_cities=args.num_cities,
            poi_key=args.poi_key,
            poi_value=args.poi_value
        )
    else:
        print(f"Неизвестный режим: {args.mode}")
        sys.exit(1)

    matrix = None
    if args.output_json or args.output_mat:
        matrix = compute_distance_matrix(points)

    def outpath(filename: str) -> str:
        return os.path.join(args.output_dir, filename)

    if args.output_csv:
        save_csv(outpath(args.output_csv), points)
    if args.output_tsp:
        save_tsp(outpath(args.output_tsp), points)
    if args.output_json:
        save_json(outpath(args.output_json), points, matrix)
    if args.output_mat:
        save_mat(outpath(args.output_mat), matrix)
    if args.output_png:
        visualize(outpath(args.output_png), points, args.connect)

if __name__ == '__main__':
    main()
