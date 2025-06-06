import random
from typing import List, Tuple

def generate_uniform(
    n: int,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    seed: int = 0
) -> List[Tuple[float, float]]:
    rnd = random.Random(seed)
    return [
        (rnd.uniform(x_min, x_max), rnd.uniform(y_min, y_max))
        for _ in range(n)
    ]
