import random
from typing import List, Tuple

def generate_cluster(
    n: int,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    clusters: int,
    std: float,
    seed: int = 0
) -> List[Tuple[float, float]]:
    rnd = random.Random(seed)
    # Случайные центры
    centers = [
        (rnd.uniform(x_min, x_max), rnd.uniform(y_min, y_max))
        for _ in range(clusters)
    ]
    points: List[Tuple[float, float]] = []
    for i in range(n):
        cx, cy = centers[rnd.randrange(clusters)]
        x = rnd.gauss(cx, std)
        y = rnd.gauss(cy, std)
        # Обрезаем по границам
        x = min(max(x, x_min), x_max)
        y = min(max(y, y_min), y_max)
        points.append((x, y))
    return points
