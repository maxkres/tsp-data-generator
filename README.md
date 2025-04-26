# TSP Data Generator

Инструмент для генерации тестовых наборов точек (городов) для классической евклидовой задачи коммивояжёра. Поддерживает разные режимы распределения, экспорт в несколько форматов и простую визуализацию.

## Возможности

- **Режимы распределения**  
  - `uniform` – равномерное по прямоугольнику  
  - `cluster` – кластеры с гауссовским разбросом  
  - `circle` – внутри или строго на границе круга  
- **Форматы вывода**  
  - CSV (`index,x,y`)  
  - TSPLIB (`.tsp`)  
  - JSON (точки + матрица расстояний)  
  - Матрица расстояний (`.npy` или `.csv`)  
  - PNG-визуализация (scatter, опция `--connect` для чернового тура)  
- Параметр `--output-dir` для указания папки вывода  
- Повторяемость через `--seed`  

## Установка

1. Клонируйте репозиторий и перейдите в него:
   ```bash
   git clone https://github.com/yourname/tsp-data-generator.git
   cd tsp-data-generator
   ```
2. Установите зависимости:
   ```bash
   pip install numpy matplotlib
   ```

## Запуск

В корне проекта есть скрипт `generate.py` — его и запускаем:

```bash
python generate.py \
  --mode uniform \
  --num-cities 50 --seed 123 \
  --output-csv uniform.csv \
  --output-tsp uniform.tsp \
  --output-json uniform.json \
  --output-mat uniform.npy \
  --output-png uniform.png --connect
```

По умолчанию все файлы появятся в папке `output/`, которая создаётся автоматически.

## Примеры использования

```bash
# 1) Cluster + JSON + CSV-матрица
python generate.py \
  --mode cluster \
  --num-cities 30 --seed 42 \
  --clusters 4 --cluster-std 0.1 \
  --output-json cluster.json \
  --output-mat cluster.csv

# 2) Круг строго по границе, только PNG
python generate.py \
  --mode circle \
  --num-cities 20 --seed 99 \
  --radius 2.5 --on-boundary \
  --output-png circle.png

# 3) Сохранить uniform в свою папку results
python generate.py \
  --mode uniform \
  --num-cities 10 --seed 7 \
  --output-csv u2.csv --output-tsp u2.tsp \
  --output-dir results
```

## Требования

- Python 3.6+
- numpy
- matplotlib
