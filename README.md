<<<<<<< HEAD

# AdaptiveTSP

Краткий TSP-решатель, основанный на адаптивной сетке (SOM) и итеративном улучшении с помощью 2-opt.

## Установка

1. Клонировать репозиторий и перейти в папку проекта:
   ```bash
   git clone https://github.com/maxkres/tsp-project.git
   cd tsp-project


2. Установить зависимости:

   ```bash
   pip install -r requirements.txt
   ```
3. Установить сам пакет:

   ```bash
   pip install .
   ```

## Использование (CLI)

* **Генерация точек** (пример):

  ```bash
  tsp-gen --type uniform --n 50 --output points.csv
  ```

  Формат CSV:

  ```
  index,x,y
  0,12.34,56.78
  1,23.45,67.89
  ...
  ```

* **Решение TSP**:

  ```bash
  adapt-tsp --input-csv points.csv --output-tour tour.csv --iterations 1000
  ```

  * `--input-csv` — входной файл с координатами точек.
  * `--output-tour` — файл, куда будет сохранён оптимизированный маршрут (в порядке обхода индексов).
  * `--iterations` — число итераций обучения SOM (по умолчанию 1000).

После выполнения в консоли выводятся длина исходного тура и длина после 2-opt.

## Пример использования (Python API)

```python
from adaptive_tsp.adaptive_tsp import AdaptiveTSP

# Список координат: [(x1, y1), (x2, y2), ...]
points = [(12.34, 56.78), (23.45, 67.89), …]

# Создаём объект-решатель (1000 итераций SOM)
solver = AdaptiveTSP(points, iterations=1000)

# Получаем маршрут (список индексов точек в порядке обхода)
tour = solver.solve()

# Сохраняем тур в CSV (опционально)
solver.save_tour("tour.csv")

# Узнаём длины:
init_len, final_len = solver.get_lengths()
print(f"Начальная длина: {init_len:.6f}")
print(f"Итоговая длина: {final_len:.6f}")
```

## Тестирование

Запустить все модульные тесты:

```bash
pytest tests
```

Файлы тестов:

* `tests/test_adaptive_grid.py`
* `tests/test_adaptive_tsp.py`
* `tests/test_utils.py`
```
=======

# AdaptiveTSP

Краткий TSP-решатель, основанный на адаптивной сетке (SOM) и итеративном улучшении с помощью 2-opt.

## Установка

1. Клонировать репозиторий и перейти в папку проекта:
   ```bash
   git clone https://github.com/maxkres/tsp-project.git
   cd tsp-project


2. Установить зависимости:

   ```bash
   pip install -r requirements.txt
   ```
3. Установить сам пакет:

   ```bash
   pip install .
   ```

## Использование (CLI)

* **Генерация точек** (пример):

  ```bash
  tsp-gen --type uniform --n 50 --output points.csv
  ```

  Формат CSV:

  ```
  index,x,y
  0,12.34,56.78
  1,23.45,67.89
  ...
  ```

* **Решение TSP**:

  ```bash
  adapt-tsp --input-csv points.csv --output-tour tour.csv --iterations 1000
  ```

  * `--input-csv` — входной файл с координатами точек.
  * `--output-tour` — файл, куда будет сохранён оптимизированный маршрут (в порядке обхода индексов).
  * `--iterations` — число итераций обучения SOM (по умолчанию 1000).

После выполнения в консоли выводятся длина исходного тура и длина после 2-opt.

## Пример использования (Python API)

```python
from adaptive_tsp.adaptive_tsp import AdaptiveTSP

# Список координат: [(x1, y1), (x2, y2), ...]
points = [(12.34, 56.78), (23.45, 67.89), …]

# Создаём объект-решатель (1000 итераций SOM)
solver = AdaptiveTSP(points, iterations=1000)

# Получаем маршрут (список индексов точек в порядке обхода)
tour = solver.solve()

# Сохраняем тур в CSV (опционально)
solver.save_tour("tour.csv")

# Узнаём длины:
init_len, final_len = solver.get_lengths()
print(f"Начальная длина: {init_len:.6f}")
print(f"Итоговая длина: {final_len:.6f}")
```

## Тестирование

Запустить все модульные тесты:

```bash
pytest tests
```

Файлы тестов:

* `tests/test_adaptive_grid.py`
* `tests/test_adaptive_tsp.py`
* `tests/test_utils.py`
```
>>>>>>> fb8bf9dc1b446bcb84e2d24c82787821568983df
