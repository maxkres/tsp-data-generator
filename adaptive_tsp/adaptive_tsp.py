import os
import csv
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from .adaptive_grid import AdaptiveGrid

class AdaptiveTSP:
    def __init__(
        self,
        points,
        iterations: int = 1000,
        som_nodes: int = None,
        som_lr_initial: float = 0.8,
        som_lr_final: float = 0.01,
        som_radius_final: float = None
    ):
        self.points = list(points)
        self.n_cities = len(self.points)
        self.iterations = iterations

        if som_nodes is None:
            self.som_nodes = 2 * self.n_cities
        else:
            self.som_nodes = som_nodes

        self.som_lr_initial = som_lr_initial
        self.som_lr_final = som_lr_final

        xs, ys = zip(*self.points)
        dx = max(xs) - min(xs)
        dy = max(ys) - min(ys)
        if som_radius_final is None:
            self.som_radius_final = math.hypot(dx, dy) / 2.0
        else:
            self.som_radius_final = som_radius_final

        self.distance_matrix = self._compute_distance_matrix(self.points)

        self.initial_tour = None
        self.improved_tour = None
        self.initial_length = None
        self.final_length = None
        self.viz_dir = None

    def _compute_distance_matrix(self, pts):
        n = len(pts)
        mat = [[0.0] * n for _ in range(n)]
        for i, (xi, yi) in enumerate(pts):
            for j in range(i + 1, n):
                xj, yj = pts[j]
                d = math.hypot(xi - xj, yi - yj)
                mat[i][j] = mat[j][i] = d
        return mat

    def solve(self, visualize_steps=False, viz_dir=None):
        if visualize_steps and viz_dir:
            self.viz_dir = viz_dir
            os.makedirs(self.viz_dir, exist_ok=True)

        cities_np = np.array(self.points)
        init_coords = self._init_som_nodes_circle(self.som_nodes, self.points)
        if visualize_steps:
            fname_init = os.path.join(self.viz_dir, "1_som_init.png")
            self._plot_som_init(fname_init, self.points, init_coords)

        grid_input = cities_np.tolist()
        som = AdaptiveGrid(
            points=grid_input,
            initial_nodes=self.som_nodes,
            iterations=self.iterations,
            lr_initial=self.som_lr_initial,
            lr_final=self.som_lr_final,
            radius_final=self.som_radius_final
        )
        som.train()

        trained_coords = som.get_nodes()
        history = som.get_history()

        if visualize_steps:
            fname_decay = os.path.join(self.viz_dir, "2_som_decay.png")
            self._plot_som_decay(fname_decay, history['lr_history'], history['radius_history'])

            fname_err_curve = os.path.join(self.viz_dir, "3_som_error_curve.png")
            self._plot_error_curve(fname_err_curve, history['error_history'])

            for (iter_idx, coords_snapshot) in history['history_nodes']:
                fname_snap = os.path.join(self.viz_dir, f"4_som_snap_iter{iter_idx:04d}.png")
                self._plot_som_trained_snapshot(fname_snap, self.points, coords_snapshot, iter_idx)

        node_city_map, unassigned = self._map_nodes_to_tour(self.points, trained_coords)
        raw_tour = node_city_map + unassigned

        if visualize_steps:
            fname_tour = os.path.join(self.viz_dir, "5_som_tour.png")
            self._plot_som_tour(fname_tour, self.points, trained_coords, node_city_map, unassigned)

        self.initial_tour = raw_tour[:]
        self.initial_length = self._tour_length(raw_tour)

        improved = self._two_opt(raw_tour)
        improved_len = self._tour_length(improved)

        self.improved_tour = improved[:]
        self.final_length = improved_len

        if visualize_steps:
            fname_imp = os.path.join(self.viz_dir, "6_improved_tour.png")
            self._plot_improved_tour(fname_imp, self.points, trained_coords, improved)

        return self.improved_tour

    def get_lengths(self):
        return (self.initial_length, self.final_length)

    def save_tour(self, path):
        with open(path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['pos', 'city'])
            for idx, city in enumerate(self.improved_tour, start=1):
                writer.writerow([idx, city])

    def _init_som_nodes_circle(self, n_nodes, cities_list):
        xs, ys = zip(*cities_list)
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        cx = 0.5 * (x_min + x_max)
        cy = 0.5 * (y_min + y_max)
        radius = 0.5 * math.hypot(x_max - x_min, y_max - y_min)
        coords = []
        for i in range(n_nodes):
            theta = 2 * math.pi * i / n_nodes
            coords.append((cx + radius * math.cos(theta), cy + radius * math.sin(theta)))
        return coords

    def _map_nodes_to_tour(self, cities_list, nodes, max_dist=None):
        tree = KDTree(cities_list)
        assigned = set()
        ordered = []
        for nx, ny in nodes:
            dist, city_idx = tree.query((nx, ny))
            if max_dist is not None and dist > max_dist:
                continue
            if city_idx not in assigned:
                ordered.append(city_idx)
                assigned.add(city_idx)
        unassigned = [i for i in range(len(cities_list)) if i not in assigned]
        return ordered, unassigned

    def _tour_length(self, tour):
        total = 0.0
        for i in range(len(tour)):
            a = tour[i]
            b = tour[(i + 1) % len(tour)]
            total += self.distance_matrix[a][b]
        return total

    def _two_opt(self, tour):
        improved = True
        best = tour[:]
        best_len = self._tour_length(best)
        n = len(best)
        while improved:
            improved = False
            for i in range(n - 1):
                for j in range(i + 2, n + (i > 0) - 1):
                    new_tour = best[:]
                    new_tour[i + 1:j + 1] = reversed(best[i + 1:j + 1])
                    new_len = self._tour_length(new_tour)
                    if new_len < best_len:
                        best = new_tour
                        best_len = new_len
                        improved = True
            tour = best[:]
        return best

    def _plot_som_init(self, fname, cities_list, init_coords):
        xs_c, ys_c = zip(*cities_list)
        xs_n, ys_n = zip(*init_coords)
        plt.figure(figsize=(6, 6))
        plt.scatter(xs_c, ys_c, s=20, color='black', label='Города')
        plt.scatter(xs_n, ys_n, s=30, color='red', label='Инициализация SOM')
        plt.title("SOM узлы (init)")
        plt.axis('equal')
        plt.legend(loc='upper right')
        plt.tight_layout()
        plt.savefig(fname)
        plt.close()

    def _plot_som_decay(self, fname, lr_history, radius_history):
        iters = list(range(len(lr_history)))
        plt.figure(figsize=(6, 4))
        plt.plot(iters, lr_history, label='learning rate')
        plt.plot(iters, radius_history, label='radius')
        plt.xlabel("Итерация SOM")
        plt.ylabel("Значение")
        plt.title("Убывание LR и Radius по итерациям")
        plt.legend()
        plt.tight_layout()
        plt.savefig(fname)
        plt.close()

    def _plot_error_curve(self, fname, error_history):
        iters = list(range(len(error_history)))
        plt.figure(figsize=(6, 4))
        plt.plot(iters, error_history, color='tab:blue')
        plt.xlabel("Итерация SOM")
        plt.ylabel("Средняя ошибка (расстояние узел→город)")
        plt.title("Ошибка привязки SOM по итерациям")
        plt.tight_layout()
        plt.savefig(fname)
        plt.close()

    def _plot_som_trained_snapshot(self, fname, cities_list, coords_snapshot, iteration):
        xs_c, ys_c = zip(*cities_list)
        xs_n, ys_n = zip(*coords_snapshot)
        plt.figure(figsize=(6, 6))
        plt.scatter(xs_c, ys_c, s=20, color='black', label='Города')
        plt.scatter(xs_n, ys_n, s=30, color='red', label=f'SOM на итерации {iteration}')
        plt.title(f"SOM Snapshot (iter={iteration})")
        plt.axis('equal')
        plt.legend(loc='upper right')
        plt.tight_layout()
        plt.savefig(fname)
        plt.close()

    def _plot_som_tour(self, fname, cities_list, trained_coords, node_city_map, unassigned):
        xs_c, ys_c = zip(*cities_list)
        xs_n, ys_n = zip(*trained_coords)

        plt.figure(figsize=(6, 6))
        plt.scatter(xs_c, ys_c, s=20, color='black', label='Города')
        plt.scatter(xs_n, ys_n, s=30, color='red', label='Узлы SOM (trained)')

        route_coords = []
        for idx_city in node_city_map:
            city_xy = cities_list[idx_city]
            min_d = None
            chosen_node = None
            for nx, ny in trained_coords:
                d = math.hypot(city_xy[0] - nx, city_xy[1] - ny)
                if min_d is None or d < min_d:
                    min_d = d
                    chosen_node = (nx, ny)
            route_coords.append(chosen_node)

        if route_coords:
            xs_r, ys_r = zip(*route_coords)
            plt.plot(xs_r, ys_r, '-', color='blue', linewidth=0.8, label='Тур по SOM')

        if unassigned:
            xs_u = [cities_list[i][0] for i in unassigned]
            ys_u = [cities_list[i][1] for i in unassigned]
            plt.scatter(xs_u, ys_u, s=40, marker='x', c='yellow', label='Непривязанные')

        plt.title("SOM Tour")
        plt.axis('equal')
        plt.legend(loc='upper right')
        plt.tight_layout()
        plt.savefig(fname)
        plt.close()

    def _plot_improved_tour(self, fname, cities_list, trained_coords, tour):
        xs_c, ys_c = zip(*cities_list)
        plt.figure(figsize=(6, 6))
        plt.scatter(xs_c, ys_c, s=20, color='black', label='Города')

        coords = [cities_list[i] for i in tour]
        xs_r, ys_r = zip(*coords)
        plt.plot(xs_r, ys_r, '-', color='blue', linewidth=0.8, label='Улучшенный тур')
        plt.plot([xs_r[-1], xs_r[0]], [ys_r[-1], ys_r[0]], '-', color='blue', linewidth=0.8)

        xs_n, ys_n = zip(*trained_coords)
        plt.scatter(xs_n, ys_n, s=30, color='red', label='Узлы SOM (final)')

        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title("Final Improved Tour")
        plt.axis('equal')
        plt.legend(loc='upper right')
        plt.tight_layout()
        plt.savefig(fname)
        plt.close()
