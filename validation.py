"""Validation examples. Choose MODE and edit the corresponding input paths."""
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Geod, Transformer
from scipy.spatial import cKDTree
from scipy.stats import linregress

MODE = "counts"  # counts / detection / capacity / comparison / temporal
INVENTORY = Path(__file__).with_name("COWTD.csv")
OUTPUT = Path("runs/validation")
YEARS = [2014, 2017, 2020, 2021, 2022, 2023, 2024]
CAPACITY = "data/province_capacity.csv"  # province,2014,2017,2020,2024
CAPACITY_TO_MW = 10  # Source units of 10,000 kW; use 1 if already MW.
OTHER = "data/other_inventory.csv"  # WGS84 lon,lat; apply a common study boundary first.
OTHER_REFERENCE_DATE = "set the source version/date"
REVIEW = "data/temporal_review.csv"  # year,review_year; blank review_year = not reviewed.
MODEL = "models/best.pt"
DATA = "data/data.yaml"
SPLIT = "val"  # Select threshold on val; evaluate the held-out test separately.
DEVICE = 0


def count_capacity(inventory, capacity, factor=10, years=(2014, 2017, 2020, 2024)):
    if capacity.province.duplicated().any():
        raise ValueError("Capacity must have one row per province")
    rows = []
    for year in years:
        counts = inventory[inventory.year <= year].groupby("province").size()
        x = capacity.province.map(counts).fillna(0).to_numpy()
        y = capacity[str(year)].to_numpy(dtype=float) * factor
        if not np.isfinite(y).all():
            raise ValueError("Missing/nonfinite province capacity")
        fit = linregress(x, y)
        rows.append({"year": year, "n_provinces": len(x), "r2": fit.rvalue**2,
                     "slope_mw": fit.slope, "intercept_mw": fit.intercept})
    return pd.DataFrame(rows)


def spatial_agreement(ours, other, radius=100):
    """Many-to-one matches within a WGS84 geodesic radius; denominator is ours."""
    if ours.empty or radius <= 0:
        raise ValueError("Nonempty input and positive radius required")
    transform = Transformer.from_crs(4326, 4978, always_xy=True)
    def xyz(frame):
        coords = frame[["lon", "lat"]].to_numpy(dtype=float)
        if not np.isfinite(coords).all() or (np.abs(coords) > [180, 90]).any():
            raise ValueError("Expected finite WGS84 longitude/latitude")
        return np.column_stack(transform.transform(coords[:, 0], coords[:, 1], np.zeros(len(frame))))
    matched = np.zeros(len(ours), dtype=bool)
    if len(other):
        candidates = cKDTree(xyz(other)).query_ball_point(xyz(ours), radius + .01)
        geod = Geod(ellps="WGS84")
        for i, js in enumerate(candidates):
            if js:
                a, b = ours.iloc[i], other.iloc[js]
                _, _, distances = geod.inv(np.full(len(js), a.lon), np.full(len(js), a.lat),
                                            b.lon.to_numpy(), b.lat.to_numpy())
                matched[i] = np.min(distances) <= radius
    return matched


if __name__ == "__main__":
    OUTPUT.mkdir(parents=True, exist_ok=False)
    if MODE == "detection":
        from ultralytics import YOLO
        result = YOLO(MODEL).val(data=DATA, split=SPLIT, imgsz=256, device=DEVICE,
                                plots=True, project=str(OUTPUT), name=SPLIT)
        print(result.results_dict)
        # Inspect the validation F1-confidence curve to select the inference threshold.
    elif MODE == "temporal":
        review = pd.read_csv(REVIEW)
        valid = review.review_year.notna()
        actual, assigned = review.loc[valid, "review_year"], review.loc[valid, "year"]
        if not actual.isin(YEARS).all() or not assigned.isin(YEARS).all():
            raise ValueError("Review years must be sampled observation years")
        steps = {year: i for i, year in enumerate(YEARS)}
        delta = actual.map(steps).to_numpy() - assigned.map(steps).to_numpy()
        pd.DataFrame([{"reviewed": len(delta), "agreement": np.mean(delta == 0) if len(delta) else np.nan,
                       "within_one_observation_step": np.mean(abs(delta) <= 1) if len(delta) else np.nan}]).to_csv(OUTPUT / "temporal.csv", index=False)
    else:
        data = pd.read_csv(INVENTORY)
        if not data.year.isin(YEARS).all():
            raise ValueError("Invalid/unassigned inventory year")
        if MODE == "counts":
            rows = [{"year": year, "count": int((data.year <= year).sum())} for year in YEARS]
            pd.DataFrame(rows).to_csv(OUTPUT / "national_counts.csv", index=False)
        elif MODE == "capacity":
            count_capacity(data, pd.read_csv(CAPACITY), CAPACITY_TO_MW).to_csv(OUTPUT / "count_capacity_r2.csv", index=False)
        elif MODE == "comparison":
            matched = spatial_agreement(data, pd.read_csv(OTHER))
            data.assign(matched=matched).to_csv(OUTPUT / "matches.csv", index=False)
            pd.DataFrame([{"reference_date": OTHER_REFERENCE_DATE, "radius_m": 100,
                           "denominator": len(data), "matched": int(matched.sum()),
                           "percentage": matched.mean() * 100}]).to_csv(OUTPUT / "agreement.csv", index=False)
        else:
            raise ValueError(f"Unknown MODE: {MODE}")
