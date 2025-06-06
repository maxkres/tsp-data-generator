# File: data_generator/src/tsp_generator/visualize.py

import matplotlib.pyplot as plt
from typing import List, Tuple

def visualize(path: str, points: List[Tuple[float, float]], connect: bool):
    xs, ys = zip(*points)
    plt.figure(figsize=(6, 6))
    plt.scatter(xs, ys, s=20, color='blue', edgecolors='black')
    if connect:
        plt.plot(xs, ys, '-', linewidth=0.8, color='gray')
        plt.plot([xs[-1], xs[0]], [ys[-1], ys[0]], '-', linewidth=0.8, color='gray')
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
