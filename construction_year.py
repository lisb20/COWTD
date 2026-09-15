"""Estimate construction years from clustered detections in local image tiles."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.spatial import cKDTree

YEARS = [2014, 2017, 2020, 2021, 2022, 2023, 2024]
PROVINCES = ["hebei"]  # Set the province list; its order determines the merge order.
LABEL_ROOT = Path("runs/predict")  # {year}/{province}/{row}_{col}.txt
CLUSTER_ROOT = Path("runs/clusters")  # {province}/clusters.json, from 2024.
OUTPUT = Path("runs/inventory.csv")
ZOOM = 17
MERGE_RADIUS = 50  # Distance in EPSG:3857 projected coordinates.
YEAR_RADIUS = 100


def read_points(year, provinces, label_root, cluster_root):
    rows = []
    for province in provinces:
        clusters = json.loads((cluster_root / province / "clusters.json").read_text())
        for cluster, tiles in clusters.items():
            for row, col in tiles:
                path = label_root / str(year) / province / f"{row}_{col}.txt"
                if not path.exists():
                    continue
                for line in path.read_text().splitlines():
                    if not line.strip():
                        continue
                    v = list(map(float, line.split()))
                    if len(v) not in (5, 6) or v[0] != 0 or not np.isfinite(v).all():
                        raise ValueError(f"Invalid YOLO label: {path}")
                    if not all(0 <= x <= 1 for x in v[1:]) or min(v[3:5]) <= 0:
                        raise ValueError(f"Invalid normalized coordinates/score: {path}")
                    lon = (col + v[1]) / 2**ZOOM * 360 - 180
                    lat = np.degrees(np.arctan(np.sinh(np.pi * (1 - 2 * (row + v[2]) / 2**ZOOM))))
                    rows.append((lon, lat, province, cluster))
    return pd.DataFrame(rows, columns=["lon", "lat", "province", "cluster"])


def reconstruct(observations, merge_radius=MERGE_RADIUS, year_radius=YEAR_RADIUS):
    latest = observations[max(observations)]
    if latest.empty:
        raise ValueError("No reference-year detections")
    forward = Transformer.from_crs(4326, 3857, always_xy=True)
    inverse = Transformer.from_crs(3857, 4326, always_xy=True)

    def project(frame):
        return np.column_stack(forward.transform(frame.lon.to_numpy(), frame.lat.to_numpy()))

    xy = project(latest)
    tree = cKDTree(xy)
    used, centers, seeds = set(), [], []
    for i, point in enumerate(xy):
        if i in used:
            continue
        neighbors = tree.query_ball_point(point, merge_radius)
        # Merge in input order; include all radius neighbours in the mean,
        # including those visited by an earlier seed.
        used.update(neighbors)
        centers.append(xy[neighbors].mean(axis=0))
        seeds.append(i)
    centers = np.asarray(centers)
    years = np.zeros(len(centers), dtype=int)
    for year in sorted(observations):
        if observations[year].empty:
            continue
        distance, _ = cKDTree(project(observations[year])).query(centers)
        years[(years == 0) & (distance < year_radius)] = year
    lon, lat = inverse.transform(centers[:, 0], centers[:, 1])
    seed_rows = latest.iloc[seeds]
    return pd.DataFrame({
        "turbine_id": [f"example-{i:06d}" for i in range(1, len(seeds) + 1)],
        "osm_matched": pd.array([pd.NA] * len(seeds), dtype="Int64"),  # No OSM comparison in this example.
        "score": np.nan,  # Historical 5-column labels do not contain confidence.
        "year": years, "lon": lon, "lat": lat,
        "province": seed_rows.province.to_numpy(),
    })


if __name__ == "__main__":
    observations = {year: read_points(year, PROVINCES, LABEL_ROOT, CLUSTER_ROOT) for year in YEARS}
    result = reconstruct(observations)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("x") as stream:
        result.to_csv(stream, index=False)
    print(f"Records: {len(result)}; unassigned years: {(result.year == 0).sum()}")
