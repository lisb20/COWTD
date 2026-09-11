"""Cluster nonempty reference-year tiles within ONE province; export tile groups."""
import json
from pathlib import Path

import pandas as pd
from sklearn.cluster import DBSCAN

LABELS = "runs/predict/2024/hebei"
OUTPUT = "runs/clusters/hebei"
EPS = 20  # Distance in zoom-17 tile indices, NOT metres.
MIN_SAMPLES = 10  # Number of tiles, NOT number of turbines.


def cluster_tiles(labels, output, eps=EPS, min_samples=MIN_SAMPLES):
    tiles = []
    for path in sorted(Path(labels).rglob("*.txt")):
        if path.read_text().strip():  # Empty inference labels are not detections.
            tiles.append(tuple(map(int, path.stem.split("_"))))  # row, col
    if len(tiles) != len(set(tiles)):
        raise ValueError("Duplicate row_col tile filenames")
    groups = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(tiles) if tiles else []
    retained = {}
    for tile, group in zip(tiles, groups):
        if group >= 0:
            retained.setdefault(str(int(group)), []).append(list(tile))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "clusters.json").write_text(json.dumps(retained, indent=2))
    pd.DataFrame([(r, c, int(g)) for (r, c), g in zip(tiles, groups)],
                 columns=["row", "col", "cluster"]).to_csv(output / "tile_clusters.csv", index=False)
    print(f"Tiles: {len(tiles)}; retained: {sum(len(v) for v in retained.values())}")


if __name__ == "__main__":
    cluster_tiles(LABELS, OUTPUT)
