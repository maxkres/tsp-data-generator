import random
import math
from typing import List, Tuple
from scipy.spatial import KDTree

class AdaptiveGrid:
    def __init__(
        self,
        points: List[Tuple[float, float]],
        initial_nodes: int = None,
        iterations: int = 1000,
        lr_initial: float = 0.8,
        lr_final: float = 0.005,
        radius_initial: float = None,
        radius_final: float = 1.0,
        add_threshold: float = None,
        remove_threshold: float = None
    ):
        self.points = points
        self.n = len(points)
        self.iterations = iterations
        self.lr_initial = lr_initial
        self.lr_final = lr_final
        self.radius_initial = radius_initial
        self.radius_final = radius_final
        self.add_threshold = add_threshold
        self.remove_threshold = remove_threshold

        self.tree = KDTree(self.points)
        self.m = initial_nodes if initial_nodes is not None else self.n

        cx = sum(x for x, _ in points) / self.n
        cy = sum(y for _, y in points) / self.n
        radius = self._bounding_box_radius()
        self.nodes = [
            (cx + radius * math.cos(2 * math.pi * i / self.m),
             cy + radius * math.sin(2 * math.pi * i / self.m))
            for i in range(self.m)
        ]
        self.last_bmu = [0] * self.m

        if self.radius_initial is None:
            self.radius_initial = self.m / 2.0
        if self.add_threshold is None:
            self.add_threshold = radius * 2.0 / self.m
        if self.remove_threshold is None:
            self.remove_threshold = self.iterations / self.m

        self.history_nodes = []
        self.error_history = []
        self.lr_history = []
        self.radius_history = []

    def _bounding_box_radius(self) -> float:
        xs = [x for x, _ in self.points]
        ys = [y for _, y in self.points]
        dx = max(xs) - min(xs)
        dy = max(ys) - min(ys)
        return math.hypot(dx, dy) / 2.0

    def _decay(self, initial: float, final: float, t: int) -> float:
        return initial + (final - initial) * (t / (self.iterations - 1))

    def train(self):
        self._record_snapshot(0)

        for t in range(self.iterations):
            px, py = random.choice(self.points)
            bmu_idx = self._find_bmu(px, py)

            for i in range(self.m):
                if i == bmu_idx:
                    self.last_bmu[i] = 0
                else:
                    self.last_bmu[i] += 1

            lr_t = self._decay(self.lr_initial, self.lr_final, t)
            radius_t = self._decay(self.radius_initial, self.radius_final, t)
            radius_sq = radius_t * radius_t

            self.lr_history.append(lr_t)
            self.radius_history.append(radius_t)

            for i in range(self.m):
                node_x, node_y = self.nodes[i]
                dist_sq = (node_x - px) ** 2 + (node_y - py) ** 2
                if dist_sq <= radius_sq:
                    theta = math.exp(-dist_sq / (2 * radius_sq))
                    new_x = node_x + lr_t * theta * (px - node_x)
                    new_y = node_y + lr_t * theta * (py - node_y)
                    self.nodes[i] = (new_x, new_y)

            bmu_x, bmu_y = self.nodes[bmu_idx]
            if math.hypot(bmu_x - px, bmu_y - py) > self.add_threshold:
                self._add_node(bmu_idx, (px, py))

            for i in reversed(range(self.m)):
                if self.last_bmu[i] > self.remove_threshold and self.m > 3:
                    self._remove_node(i)

            avg_err = self._average_node_error()
            self.error_history.append(avg_err)

            if (t + 1) in {
                int(self.iterations * 0.25),
                int(self.iterations * 0.50),
                int(self.iterations * 0.75),
                self.iterations - 1
            }:
                self._record_snapshot(t + 1)

    def _find_bmu(self, px: float, py: float) -> int:
        best_idx = 0
        best_dist = float('inf')
        for i, (nx, ny) in enumerate(self.nodes):
            d = (nx - px) ** 2 + (ny - py) ** 2
            if d < best_dist:
                best_dist = d
                best_idx = i
        return best_idx

    def _add_node(self, bmu_idx: int, point: Tuple[float, float]):
        insert_idx = bmu_idx + 1
        self.nodes.insert(insert_idx, point)
        self.last_bmu.insert(insert_idx, 0)
        self.m += 1

    def _remove_node(self, idx: int):
        self.nodes.pop(idx)
        self.last_bmu.pop(idx)
        self.m -= 1

    def _average_node_error(self) -> float:
        if self.m == 0:
            return 0.0
        dists, _ = self.tree.query(self.nodes)
        return float(sum(dists) / len(dists))

    def _record_snapshot(self, iteration: int):
        coords_copy = [(x, y) for x, y in self.nodes]
        self.history_nodes.append((iteration, coords_copy))

    def get_nodes(self) -> List[Tuple[float, float]]:
        return self.nodes.copy()

    def get_history(self):
        return {
            'history_nodes': self.history_nodes,
            'error_history': self.error_history,
            'lr_history': self.lr_history,
            'radius_history': self.radius_history
        }
