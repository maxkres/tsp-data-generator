import os
import argparse

from tsp_generator.generators.uniform import generate_uniform
from tsp_generator.generators.cluster import generate_cluster
from tsp_generator.generators.circle import generate_circle

from tsp_generator.output import (
    save_csv, save_tsp, compute_distance_matrix,
    save_json, save_mat
)
from tsp_generator.visualize import visualize

def main():
    parser = argparse.ArgumentParser(
        description="Генератор данных для евклидового TSP"
    )
    parser.add_argument('--mode', choices=['uniform', 'cluster', 'circle'],
                        default='uniform', help='Режим генерации')
    parser.add_argument('-n', '--num-cities', type=int, default=100,
                        help='Число городов')
    parser.add_argument('--x-min', type=float, default=0.0)
    parser.add_argument('--x-max', type=float, default=1.0)
    parser.add_argument('--y-min', type=float, default=0.0)
    parser.add_argument('--y-max', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=0)

    parser.add_argument('--clusters', type=int, default=5,
                        help='Число кластеров (mode=cluster)')
    parser.add_argument('--cluster-std', type=float, default=0.05,
                        help='Sigma разброса (mode=cluster)')

    parser.add_argument('--radius', type=float, default=1.0,
                        help='Радиус круга (mode=circle)')
    parser.add_argument('--on-boundary', action='store_true',
                        help='Только на границе круга (mode=circle)')

    parser.add_argument('--output-csv', type=str,
                        help='Сохранить CSV (index,x,y)')
    parser.add_argument('--output-tsp', type=str,
                        help='Сохранить TSPLIB .tsp')
    parser.add_argument('--output-json', type=str,
                        help='Сохранить JSON (точки + матрица)')
    parser.add_argument('--output-mat', type=str,
                        help='Сохранить матрицу расстояний (.npy или .csv)')
    parser.add_argument('--output-png', type=str,
                        help='Сохранить scatter-plot PNG')
    parser.add_argument('--connect', action='store_true',
                        help='Соединить точки линиями в plot.png')
    parser.add_argument('--output-dir', type=str, default='output',
                        help='Папка для всех выходных файлов')

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

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
    else:
        points = generate_circle(
            args.num_cities,
            args.radius,
            args.on_boundary,
            args.seed
        )

    # Если нужен JSON или матрица, считаем её один раз
    matrix = None
    if args.output_json or args.output_mat:
        matrix = compute_distance_matrix(points)

    def outpath(filename: str) -> str:
        return os.path.join(args.output_dir, filename)

    if args.output_csv:
        dest = outpath(args.output_csv)
        save_csv(dest, points)
        print(f"✅ CSV → {dest}")

    if args.output_tsp:
        dest = outpath(args.output_tsp)
        save_tsp(dest, points)
        print(f"✅ TSPLIB → {dest}")

    if args.output_json:
        dest = outpath(args.output_json)
        save_json(dest, points, matrix)
        print(f"✅ JSON → {dest}")

    if args.output_mat:
        dest = outpath(args.output_mat)
        save_mat(dest, matrix)
        print(f"✅ Matrix → {dest}")

    if args.output_png:
        dest = outpath(args.output_png)
        visualize(dest, points, args.connect)
        print(f"✅ Plot → {dest}")

    if not any([args.output_csv, args.output_tsp,
                args.output_json, args.output_mat,
                args.output_png]):
        print("⚠️  Нечего сохранять: укажите хотя бы один --output-*")


if __name__ == '__main__':
    main()
