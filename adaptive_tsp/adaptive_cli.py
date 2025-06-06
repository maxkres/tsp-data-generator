#!/usr/bin/env python3
import os
import sys
import argparse
import csv
import json

proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if proj_root not in sys.path:
    sys.path.insert(0, proj_root)

from adaptive_tsp.adaptive_tsp import AdaptiveTSP

def main():
    parser = argparse.ArgumentParser(description="AdaptiveTSP")
    parser.add_argument('--input-csv', type=str, help='CSV: index,x,y')
    parser.add_argument('--input-json', type=str, help='JSON (points + matrix) от data_generator')
    parser.add_argument('--iterations', type=int, default=1000, help='Число итераций для обучения SOM')
    parser.add_argument('--output-tour', type=str, help='CSV-файл для сохранения финального тура (pos,city)')
    parser.add_argument('--viz-dir', type=str, default=None, help='Папка для PNG-визуализаций всех этапов (если указано)')
    args = parser.parse_args()

    if not args.input_csv and not args.input_json:
        print("Ошибка: нужно указать либо --input-csv, либо --input-json")
        sys.exit(1)

    if args.input_json:
        with open(args.input_json, 'r', encoding='utf-8') as f:
            data = json.load(f)
        points = [(pt['x'], pt['y']) for pt in data['points']]
    else:
        points = []
        with open(args.input_csv, newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                _, x, y = row
                points.append((float(x), float(y)))

    solver = AdaptiveTSP(points, iterations=args.iterations)

    if args.viz_dir:
        tour = solver.solve(visualize_steps=True, viz_dir=args.viz_dir)
    else:
        tour = solver.solve(visualize_steps=False, viz_dir=None)

    init_len, final_len = solver.get_lengths()

    print(f"Длина начального тура: {init_len:.6f}")
    print(f"Длина после 2-opt:    {final_len:.6f}")

    if args.output_tour:
        solver.save_tour(args.output_tour)
        print(f"✅ Тур сохранён в {args.output_tour}")

if __name__ == "__main__":
    main()
