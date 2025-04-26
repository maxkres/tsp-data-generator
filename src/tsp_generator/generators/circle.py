import random
import math
from typing import List, Tuple

def generate_circle(
    n: int,
    radius: float,
    on_boundary: bool,
    seed: int = 0
) -> List[Tuple[float, float]]:
    rnd = random.Random(seed)
    points: List[Tuple[float, float]] = []
    for _ in range(n):
        theta = rnd.uniform(0, 2 * math.pi)
        if on_boundary:
            r = radius
        else:
            # для равномерного в круге используем sqrt(U)
            r = math.sqrt(rnd.uniform(0, 1)) * radius
        x = r * math.cos(theta)
        y = r * math.sin(theta)
        points.append((x, y))
    return points
