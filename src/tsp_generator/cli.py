import argparse
from tsp_generator.generators.uniform import generate_uniform
from tsp_generator.generators.cluster import generate_cluster
from tsp_generator.generators.circle import generate_circle
from tsp_generator.output import save_csv, save_tsp

def main():
    p = argparse.ArgumentParser(
        description="Генератор данных для евклидового TSP"
    )
    p.add_argument('--mode', choices=['uniform', 'cluster', 'circle'],
                   default='uniform', help='Режим генерации')
    p.add_argument('-n', '--num-cities', type=int, default=100,
                   help='Число городов')
    p.add_argument('--x-min', type=float, default=0.0)
    p.add_argument('--x-max', type=float, default=1.0)
    p.add_argument('--y-min', type=float, default=0.0)
    p.add_argument('--y-max', type=float, default=1.0)
    p.add_argument('--seed', type=int, default=0,
                   help='Random seed')

    # Параметры для кластерного режима
    p.add_argument('--clusters', type=int, default=5,
                   help='Число кластеров (mode=cluster)')
    p.add_argument('--cluster-std', type=float, default=0.05,
                   help='Sigma разброса в кластере')

    # Параметры для кругового режима
    p.add_argument('--radius', type=float, default=1.0,
                   help='Радиус круга (mode=circle)')
    p.add_argument('--on-boundary', action='store_true',
                   help='Точки на границе круга')

    p.add_argument('--output-csv', type=str,
                   help='Сохранить CSV (index,x,y)')
    p.add_argument('--output-tsp', type=str,
                   help='Сохранить TSPLIB .tsp')

    args = p.parse_args()

    if args.mode == 'uniform':
        pts = generate_uniform(
            args.num_cities,
            args.x_min, args.x_max,
            args.y_min, args.y_max,
            args.seed
        )
    elif args.mode == 'cluster':
        pts = generate_cluster(
            args.num_cities,
            args.x_min, args.x_max,
            args.y_min, args.y_max,
            args.clusters, args.cluster_std,
            args.seed
        )
    else:
        pts = generate_circle(
            args.num_cities,
            args.radius,
            args.on_boundary,
            args.seed
        )

    if args.output_csv:
        save_csv(args.output_csv, pts)
        print(f"✅ CSV сохранено: {args.output_csv}")
    if args.output_tsp:
        save_tsp(args.output_tsp, pts)
        print(f"✅ TSPLIB сохранён: {args.output_tsp}")
    if not args.output_csv and not args.output_tsp:
        print("⚠️  Укажите --output-csv и/или --output-tsp")

if __name__ == '__main__':
    main()
