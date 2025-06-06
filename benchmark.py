#!/usr/bin/env python3
import os
import sys
import argparse
import time
import itertools
import math

try:
    from tqdm import tqdm
except ImportError:
    print("Требуется установить tqdm: pip install tqdm")
    sys.exit(1)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

_this_dir = os.path.dirname(__file__)
sys.path.insert(0, _this_dir)

from bench_utils.benchmark_utils import (
    generate_synthetic,
    load_osm,
    load_tsplib,
    compute_baseline_nn_2opt,
    read_optimum_tsplib
)
from adaptive_tsp.adaptive_tsp import AdaptiveTSP


def parse_list_of_ints(s: str):
    return [int(x.strip()) for x in s.split(',') if x.strip()]


def plot_synthetic_results(df: pd.DataFrame, distribution: str, n: int, output_dir: str):
    subset = df[(df['mode'] == 'synthetic') &
                (df['submode'] == distribution) &
                (df['n'] == n)]
    if subset.empty:
        return

    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(6, 4))
    for T_val in sorted(subset['T'].unique()):
        df_T = subset[subset['T'] == T_val]
        m_vals = df_T['m'].values
        means = df_T['final_len_som_mean'].values
        stds = df_T['final_len_som_std'].values
        plt.errorbar(m_vals, means, yerr=stds, marker='o', capsize=3, label=f"T={T_val}")
    plt.xlabel("m (число узлов SOM)")
    plt.ylabel("Длина тура (среднее ± std)")
    plt.title(f"{distribution.capitalize()}: длина тура vs m (n={n})")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{distribution}_n{n}_len_vs_m.png"))
    plt.close()

    plt.figure(figsize=(6, 4))
    for T_val in sorted(subset['T'].unique()):
        df_T = subset[subset['T'] == T_val]
        m_vals = df_T['m'].values
        means = df_T['time_som_total_mean'].values
        stds = df_T['time_som_total_std'].values
        plt.errorbar(m_vals, means, yerr=stds, marker='o', capsize=3, label=f"T={T_val}")
    plt.xlabel("m (число узлов SOM)")
    plt.ylabel("Время SOM (среднее ± std), сек")
    plt.title(f"{distribution.capitalize()}: время SOM vs m (n={n})")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{distribution}_n{n}_time_vs_m.png"))
    plt.close()

    plt.figure(figsize=(6, 4))
    indices = np.arange(len(subset))
    width = 0.35
    means_som = subset['final_len_som_mean'].values
    std_som = subset['final_len_som_std'].values
    means_base = subset['baseline_len_mean'].values
    std_base = subset['baseline_len_std'].values
    plt.bar(indices - width/2, means_som, width, yerr=std_som, capsize=3, label='AdaptiveTSP')
    plt.bar(indices + width/2, means_base, width, yerr=std_base, capsize=3, label='Baseline NN+2-opt')
    labels = [f"m={int(m)},T={int(T)}" for m, T in zip(subset['m'], subset['T'])]
    plt.xticks(indices, labels, rotation=45, ha='right')
    plt.xlabel("(m, T)")
    plt.ylabel("Длина тура (среднее ± std)")
    plt.title(f"{distribution.capitalize()} (n={n}): AdaptiveTSP vs Baseline")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{distribution}_n{n}_compare_baseline.png"))
    plt.close()


def plot_osm_results(df: pd.DataFrame, region: str, output_dir: str):
    subset = df[(df['mode'] == 'osm') & (df['submode'] == region)]
    if subset.empty:
        return

    os.makedirs(output_dir, exist_ok=True)
    n = int(subset['n'].iloc[0])

    plt.figure(figsize=(6, 4))
    for T_val in sorted(subset['T'].unique()):
        df_T = subset[subset['T'] == T_val]
        m_vals = df_T['m'].values
        means = df_T['final_len_som_mean'].values
        stds = df_T['final_len_som_std'].values
        plt.errorbar(m_vals, means, yerr=stds, marker='o', capsize=3, label=f"T={T_val}")
    plt.xlabel("m (число узлов SOM)")
    plt.ylabel("Длина тура (среднее ± std)")
    plt.title(f"OSM: {region} (n={n}) длина тура vs m")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"osm_{region.replace(' ', '_')}_len_vs_m.png"))
    plt.close()

    plt.figure(figsize=(6, 4))
    for T_val in sorted(subset['T'].unique()):
        df_T = subset[subset['T'] == T_val]
        m_vals = df_T['m'].values
        means = df_T['time_som_total_mean'].values
        stds = df_T['time_som_total_std'].values
        plt.errorbar(m_vals, means, yerr=stds, marker='o', capsize=3, label=f"T={T_val}")
    plt.xlabel("m (число узлов SOM)")
    plt.ylabel("Время SOM (среднее ± std), сек")
    plt.title(f"OSM: {region} (n={n}) время SOM vs m")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"osm_{region.replace(' ', '_')}_time_vs_m.png"))
    plt.close()

    plt.figure(figsize=(6, 4))
    indices = np.arange(len(subset))
    width = 0.35
    means_som = subset['final_len_som_mean'].values
    std_som = subset['final_len_som_std'].values
    means_base = subset['baseline_len_mean'].values
    std_base = subset['baseline_len_std'].values
    plt.bar(indices - width/2, means_som, width, yerr=std_som, capsize=3, label='AdaptiveTSP')
    plt.bar(indices + width/2, means_base, width, yerr=std_base, capsize=3, label='Baseline NN+2-opt')
    labels = [f"m={int(m)},T={int(T)}" for m, T in zip(subset['m'], subset['T'])]
    plt.xticks(indices, labels, rotation=45, ha='right')
    plt.xlabel("(m, T)")
    plt.ylabel("Длина тура (среднее ± std)")
    plt.title(f"OSM: {region} (n={n}) AdaptiveTSP vs Baseline")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"osm_{region.replace(' ', '_')}_compare_baseline.png"))
    plt.close()


def plot_tsplib_results(df: pd.DataFrame, filename: str, output_dir: str):
    subset = df[(df['mode'] == 'tsplib') & (df['submode'] == filename)]
    if subset.empty:
        return

    os.makedirs(output_dir, exist_ok=True)
    n = int(subset['n'].iloc[0])

    if subset['deviation_mean'].notnull().any():
        plt.figure(figsize=(6, 4))
        for T_val in sorted(subset['T'].unique()):
            df_T = subset[subset['T'] == T_val]
            m_vals = df_T['m'].values
            means = df_T['deviation_mean'].values
            stds = df_T['deviation_std'].values
            plt.errorbar(m_vals, means, yerr=stds, marker='o', capsize=3, label=f"T={T_val}")
        plt.xlabel("m (число узлов SOM)")
        plt.ylabel("Отклонение (%) (среднее ± std)")
        plt.title(f"TSPLIB: {filename} (n={n}) отклонение vs m")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"tsplib_{filename}_deviation_vs_m.png"))
        plt.close()

    plt.figure(figsize=(6, 4))
    for T_val in sorted(subset['T'].unique()):
        df_T = subset[subset['T'] == T_val]
        m_vals = df_T['m'].values
        means = df_T['final_len_som_mean'].values
        stds = df_T['final_len_som_std'].values
        plt.errorbar(m_vals, means, yerr=stds, marker='o', capsize=3, label=f"T={T_val}")
    plt.xlabel("m (число узлов SOM)")
    plt.ylabel("Длина тура (среднее ± std)")
    plt.title(f"TSPLIB: {filename} (n={n}) длина тура vs m")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"tsplib_{filename}_len_vs_m.png"))
    plt.close()

    plt.figure(figsize=(6, 4))
    for T_val in sorted(subset['T'].unique()):
        df_T = subset[subset['T'] == T_val]
        m_vals = df_T['m'].values
        means = df_T['time_som_total_mean'].values
        stds = df_T['time_som_total_std'].values
        plt.errorbar(m_vals, means, yerr=stds, marker='o', capsize=3, label=f"T={T_val}")
    plt.xlabel("m (число узлов SOM)")
    plt.ylabel("Время SOM (среднее ± std), сек")
    plt.title(f"TSPLIB: {filename} (n={n}) время SOM vs m")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"tsplib_{filename}_time_vs_m.png"))
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Запуск бенчмарков AdaptiveTSP с усреднением и визуализациями")
    parser.add_argument('--mode', type=str, required=True,
                        choices=['synthetic', 'osm', 'tsplib'],
                        help="Режим бенчмарка: synthetic | osm | tsplib")
    parser.add_argument('--distribution', type=str, default=None,
                        help="Для synthetic: тип ['uniform','ring','clusters']")
    parser.add_argument('--n', type=int, default=None,
                        help="Число городов (для synthetic и osm)")
    parser.add_argument('--clusters', type=int, default=None,
                        help="Для synthetic=clusters: число кластеров")
    parser.add_argument('--seed', type=int, default=42,
                        help="Базовый seed для генерации (synthetic)")
    parser.add_argument('--x_min', type=float, default=0.0, help="synthetic: x_min")
    parser.add_argument('--x_max', type=float, default=1.0, help="synthetic: x_max")
    parser.add_argument('--y_min', type=float, default=0.0, help="synthetic: y_min")
    parser.add_argument('--y_max', type=float, default=1.0, help="synthetic: y_max")
    parser.add_argument('--radius', type=float, default=0.5,
                        help="synthetic=ring: радиус круга")
    parser.add_argument('--on_boundary', action='store_true',
                        help="synthetic=ring: генерировать точки строго на границе")
    parser.add_argument('--std', type=float, default=0.05,
                        help="synthetic=clusters: стандартное отклонение кластеров")
    parser.add_argument('--osm_region', type=str, default=None,
                        help="Для mode=osm: название региона (например, 'Moscow, Russia')")
    parser.add_argument('--osm_poi_key', type=str, default='amenity',
                        help="OSM: ключ POI (default: 'amenity')")
    parser.add_argument('--osm_poi_value', type=str, default='restaurant',
                        help="OSM: значение POI (default: 'restaurant')")
    parser.add_argument('--tsplib_file', type=str, default=None,
                        help="Для mode=tsplib: путь к файлу .tsp")
    parser.add_argument('--m_values', type=str, required=True,
                        help="Список m (число узлов SOM), через запятую, например '100,200,400'")
    parser.add_argument('--T_values', type=str, required=True,
                        help="Список T (число итераций), через запятую, например '500,1000,2000'")
    parser.add_argument('--output_csv', type=str, required=True,
                        help="Куда сохранить CSV с результатами (путь+имя).")
    parser.add_argument('--plots_dir', type=str, default="plots",
                        help="Каталог для сохранения графиков (по умолчанию: plots).")
    parser.add_argument('--n_repeats', type=int, default=5,
                        help="Число повторений для усреднения (synthetic только).")

    args = parser.parse_args()

    m_list = parse_list_of_ints(args.m_values)
    T_list = parse_list_of_ints(args.T_values)
    base_seed = args.seed
    n_repeats = args.n_repeats

    results = []

    if args.mode == 'synthetic':
        if args.distribution not in ['uniform', 'ring', 'clusters']:
            print("❌ Для synthetic distribution нужно выбрать из ['uniform','ring','clusters']")
            sys.exit(1)
        if args.n is None:
            print("❌ Для synthetic нужно указать --n")
            sys.exit(1)

        total_runs = len(m_list) * len(T_list)
        print(f"Запускаем synthetic-бенчмарк ({args.distribution}), n={args.n}: {total_runs} комбинаций, каждая усредняется по {n_repeats} запускам")
        outer = tqdm(itertools.product(m_list, T_list), total=total_runs, desc="Benchmark synthetic")

        for m_val, T_val in outer:
            init_lens = []
            final_lens = []
            times_som = []
            baseline_lens = []
            baseline_times = []

            for rep in range(n_repeats):
                seed_rep = base_seed + rep
                gen_kwargs = {
                    'seed': seed_rep,
                    'x_min': args.x_min,
                    'x_max': args.x_max,
                    'y_min': args.y_min,
                    'y_max': args.y_max,
                }
                if args.distribution == 'ring':
                    gen_kwargs.update({
                        'radius': args.radius,
                        'on_boundary': args.on_boundary
                    })
                if args.distribution == 'clusters':
                    gen_kwargs.update({
                        'clusters': args.clusters,
                        'std': args.std
                    })

                pts = generate_synthetic(args.distribution, args.n, **gen_kwargs)

                t0 = time.time()
                solver = AdaptiveTSP(
                    points=pts,
                    iterations=T_val,
                    som_nodes=m_val
                )
                solver.solve(visualize_steps=False)
                t_som = time.time() - t0
                init_len, final_len = solver.get_lengths()

                baseline_tour, base_init, base_final, base_time = compute_baseline_nn_2opt(pts)

                init_lens.append(init_len)
                final_lens.append(final_len)
                times_som.append(t_som)
                baseline_lens.append(base_final)
                baseline_times.append(base_time)

            init_mean = float(np.mean(init_lens))
            init_std  = float(np.std(init_lens, ddof=1))
            final_mean = float(np.mean(final_lens))
            final_std  = float(np.std(final_lens, ddof=1))
            time_mean  = float(np.mean(times_som))
            time_std   = float(np.std(times_som, ddof=1))
            base_len_mean = float(np.mean(baseline_lens))
            base_len_std  = float(np.std(baseline_lens, ddof=1))
            base_time_mean= float(np.mean(baseline_times))
            base_time_std = float(np.std(baseline_times, ddof=1))

            results.append({
                'mode': 'synthetic',
                'submode': args.distribution,
                'n': args.n,
                'm': m_val,
                'T': T_val,
                'init_len_som_mean': init_mean,
                'init_len_som_std': init_std,
                'final_len_som_mean': final_mean,
                'final_len_som_std': final_std,
                'time_som_total_mean': time_mean,
                'time_som_total_std': time_std,
                'baseline_len_mean': base_len_mean,
                'baseline_len_std': base_len_std,
                'baseline_time_mean': base_time_mean,
                'baseline_time_std': base_time_std,
                'optimum': None,
                'deviation_mean': None,
                'deviation_std': None
            })

    elif args.mode == 'osm':
        if args.osm_region is None or args.n is None:
            print("❌ Для mode=osm нужно указать --osm_region и --n")
            sys.exit(1)

        print(f"Запускаем OSM-бенчмарк ({args.osm_region}), n={args.n}")
        pts = load_osm(args.osm_region, args.n, args.osm_poi_key, args.osm_poi_value)
        if len(pts) == 0:
            print(f"❌ OSM вернул пустой список для {args.osm_region}")
            sys.exit(1)

        total_runs = len(m_list) * len(T_list)
        outer = tqdm(itertools.product(m_list, T_list), total=total_runs, desc="Benchmark OSM")

        for m_val, T_val in outer:
            init_lens = []
            final_lens = []
            times_som = []
            baseline_lens = []
            baseline_times = []

            for rep in range(n_repeats):
                t0 = time.time()
                solver = AdaptiveTSP(
                    points=pts,
                    iterations=T_val,
                    som_nodes=m_val
                )
                solver.solve(visualize_steps=False)
                t_som = time.time() - t0
                init_len, final_len = solver.get_lengths()

                baseline_tour, base_init, base_final, base_time = compute_baseline_nn_2opt(pts)

                init_lens.append(init_len)
                final_lens.append(final_len)
                times_som.append(t_som)
                baseline_lens.append(base_final)
                baseline_times.append(base_time)

            init_mean = float(np.mean(init_lens))
            init_std  = float(np.std(init_lens, ddof=1))
            final_mean = float(np.mean(final_lens))
            final_std  = float(np.std(final_lens, ddof=1))
            time_mean  = float(np.mean(times_som))
            time_std   = float(np.std(times_som, ddof=1))
            base_len_mean = float(np.mean(baseline_lens))
            base_len_std  = float(np.std(baseline_lens, ddof=1))
            base_time_mean= float(np.mean(baseline_times))
            base_time_std = float(np.std(baseline_times, ddof=1))

            results.append({
                'mode': 'osm',
                'submode': args.osm_region,
                'n': len(pts),
                'm': m_val,
                'T': T_val,
                'init_len_som_mean': init_mean,
                'init_len_som_std': init_std,
                'final_len_som_mean': final_mean,
                'final_len_som_std': final_std,
                'time_som_total_mean': time_mean,
                'time_som_total_std': time_std,
                'baseline_len_mean': base_len_mean,
                'baseline_len_std': base_len_std,
                'baseline_time_mean': base_time_mean,
                'baseline_time_std': base_time_std,
                'optimum': None,
                'deviation_mean': None,
                'deviation_std': None
            })

    elif args.mode == 'tsplib':
        if args.tsplib_file is None:
            print("❌ Для mode=tsplib нужно указать --tsplib_file")
            sys.exit(1)

        print(f"Запускаем TSPLIB-бенчмарк ({args.tsplib_file})")
        pts = load_tsplib(args.tsplib_file)
        if len(pts) == 0:
            print(f"❌ Не удалось прочитать {args.tsplib_file}")
            sys.exit(1)

        optimum = read_optimum_tsplib(args.tsplib_file)
        total_runs = len(m_list) * len(T_list)
        outer = tqdm(itertools.product(m_list, T_list), total=total_runs, desc="Benchmark TSPLIB")

        for m_val, T_val in outer:
            init_lens = []
            final_lens = []
            times_som = []
            baseline_lens = []
            baseline_times = []
            deviations = []

            for rep in range(n_repeats):
                t0 = time.time()
                solver = AdaptiveTSP(
                    points=pts,
                    iterations=T_val,
                    som_nodes=m_val
                )
                solver.solve(visualize_steps=False)
                t_som = time.time() - t0
                init_len, final_len = solver.get_lengths()

                baseline_tour, base_init, base_final, base_time = compute_baseline_nn_2opt(pts)

                init_lens.append(init_len)
                final_lens.append(final_len)
                times_som.append(t_som)
                baseline_lens.append(base_final)
                baseline_times.append(base_time)

                if optimum is not None:
                    deviations.append(100.0 * (final_len - optimum) / optimum)

            init_mean = float(np.mean(init_lens))
            init_std  = float(np.std(init_lens, ddof=1))
            final_mean = float(np.mean(final_lens))
            final_std  = float(np.std(final_lens, ddof=1))
            time_mean  = float(np.mean(times_som))
            time_std   = float(np.std(times_som, ddof=1))
            base_len_mean = float(np.mean(baseline_lens))
            base_len_std  = float(np.std(baseline_lens, ddof=1))
            base_time_mean= float(np.mean(baseline_times))
            base_time_std = float(np.std(baseline_times, ddof=1))

            devo_mean = None
            devo_std  = None
            if optimum is not None and len(deviations) > 0:
                devo_mean = float(np.mean(deviations))
                devo_std  = float(np.std(deviations, ddof=1))

            results.append({
                'mode': 'tsplib',
                'submode': os.path.basename(args.tsplib_file),
                'n': len(pts),
                'm': m_val,
                'T': T_val,
                'init_len_som_mean': init_mean,
                'init_len_som_std': init_std,
                'final_len_som_mean': final_mean,
                'final_len_som_std': final_std,
                'time_som_total_mean': time_mean,
                'time_som_total_std': time_std,
                'baseline_len_mean': base_len_mean,
                'baseline_len_std': base_len_std,
                'baseline_time_mean': base_time_mean,
                'baseline_time_std': base_time_std,
                'optimum': optimum,
                'deviation_mean': devo_mean,
                'deviation_std': devo_std
            })

    else:
        print("❌ Неподдерживаемый mode. Выберите synthetic | osm | tsplib.")
        sys.exit(1)

    df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(args.output_csv) or '.', exist_ok=True)
    df.to_csv(args.output_csv, index=False)
    print(f"✅ Результаты сохранены в {args.output_csv}")

    plots_dir = args.plots_dir
    os.makedirs(plots_dir, exist_ok=True)

    if args.mode == 'synthetic':
        for n_val in sorted(df['n'].unique()):
            plot_synthetic_results(df, args.distribution, n_val, plots_dir)
    elif args.mode == 'osm':
        for region in sorted(df['submode'].unique()):
            plot_osm_results(df, region, plots_dir)
    elif args.mode == 'tsplib':
        for filename in sorted(df['submode'].unique()):
            plot_tsplib_results(df, filename, plots_dir)

    print(f"✅ Графики сохранены в '{plots_dir}'")


if __name__ == "__main__":
    main()
