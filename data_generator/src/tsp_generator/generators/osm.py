# File: data_generator/src/tsp_generator/generators/osm.py

import requests
from typing import List, Tuple


def geocode_place(place_name: str) -> Tuple[float, float, float, float]:
    url = "https://nominatim.openstreetmap.org/search"
    params = {
        "q": place_name,
        "format": "json",
        "limit": 1
    }
    headers = {
        "User-Agent": "TSPGenerator/1.0 (+https://your-domain.example)"
    }

    response = requests.get(url, params=params, headers=headers, timeout=10)
    if response.status_code != 200:
        raise RuntimeError(f"Nominatim вернул HTTP {response.status_code}")

    data = response.json()
    if not data:
        raise RuntimeError(f"Nominatim не нашёл place_name='{place_name}'")

    item = data[0]
    bbox = item.get("boundingbox", None)
    if not bbox or len(bbox) != 4:
        raise RuntimeError("Неправильный ответ от Nominatim: нет 'boundingbox'")

    lat_min = float(bbox[0])
    lat_max = float(bbox[1])
    lon_min = float(bbox[2])
    lon_max = float(bbox[3])
    return lat_min, lon_min, lat_max, lon_max


def generate_osm(
    place_name: str,
    num_cities: int,
    poi_key: str = "amenity",
    poi_value: str = "restaurant"
) -> List[Tuple[float, float]]:
    lat_min, lon_min, lat_max, lon_max = geocode_place(place_name)
    if lat_min >= lat_max or lon_min >= lon_max:
        raise RuntimeError("Некорректный bounding box от Nominatim")

    bbox_str = f"{lat_min},{lon_min},{lat_max},{lon_max}"
    overpass_url = "https://overpass-api.de/api/interpreter"
    query = f"""
    [out:json][timeout:60];
    node["{poi_key}"="{poi_value}"]({bbox_str});
    out center;
    """

    response = requests.post(overpass_url, data={"data": query}, timeout=120)
    if response.status_code != 200:
        raise RuntimeError(f"Overpass API вернул HTTP {response.status_code}")

    data = response.json()
    elements = data.get("elements", [])

    points: List[Tuple[float, float]] = []
    for el in elements:
        if el.get("type") == "node":
            lat = el.get("lat")
            lon = el.get("lon")
            if lat is not None and lon is not None:
                points.append((lat, lon))
            if len(points) >= num_cities:
                break

    if len(points) < num_cities:
        print(
            f"⚠️ Overpass вернул только {len(points)} узлов для {poi_key}={poi_value}, "
            f"хотя запрошено {num_cities}."
        )

    return points[:num_cities]
