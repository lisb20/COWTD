"""Inspect and export simulated wind power and capacity factors from NetCDF."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


def hourly_frame(ds: xr.Dataset, province: str | None = None) -> pd.DataFrame:
    """Keep source timestamps unchanged; output power in kW and CF in [0, 1]."""
    if "province" in ds.dims:
        if province is None:
            raise ValueError("Select a province with --province; use --inspect to list names.")
        ds = ds.sel(province=province)
        if ds.total_generation.attrs.get("units") != "kW":
            raise ValueError("Expected total_generation units kW.")
        names = ["mean_capacity_factor", "std_capacity_factor", "weight_capacity_factor"]
        data = {name: ds[name].values for name in names}
        data["power_kw"] = ds.total_generation.values
    else:
        if "location" in ds.dims:
            if ds.sizes["location"] != 1:
                raise ValueError("Expected one location per grid file.")
            ds = ds.isel(location=0)
        cf = ds.capacity_factor
        capacity = float(ds.capacity.item())
        # Source step1_process_plant.py converts capacity from MW to kW.
        unit = ds.capacity.attrs.get("units", "kW")
        if unit != "kW":
            raise ValueError(f"Expected grid capacity in kW, got {unit!r}.")
        data = {"capacity_factor": cf.values, "power_kw": cf.values * capacity}
    frame = pd.DataFrame(data, index=pd.DatetimeIndex(ds.time.values, name="time"))
    if len(frame) < 2 or not frame.index.is_unique or not frame.index.is_monotonic_increasing:
        raise ValueError("Expected at least two unique, increasing timestamps.")
    if not np.all(np.diff(frame.index.asi8) == pd.Timedelta(hours=1).value):
        raise ValueError("Energy conversion requires regular hourly samples.")
    return frame


def daily_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Assign 1 h samples to their timestamp date and flag incomplete days."""
    daily = frame.drop(columns="power_kw").resample("D").mean()
    daily["energy_mwh"] = frame.power_kw.resample("D").sum(min_count=1) / 1000
    daily["valid_power_hours"] = frame.power_kw.resample("D").count()
    daily["complete_day"] = daily.valid_power_hours == 24
    return daily


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="External province or single-grid NetCDF")
    parser.add_argument("--province", help="Exact province coordinate, e.g. 云南省")
    parser.add_argument("--inspect", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent / "outputs")
    args = parser.parse_args()
    with xr.open_dataset(args.input) as ds:
        if args.inspect:
            report = {
                "dimensions": dict(ds.sizes), "time_start": str(ds.time.values[0]),
                "time_end": str(ds.time.values[-1]), "attributes": dict(ds.attrs),
                "variables": {n: {"dimensions": list(v.dims), "attributes": dict(v.attrs)}
                              for n, v in ds.data_vars.items()},
                "provinces": ds.province.values.tolist() if "province" in ds.dims else [],
            }
            print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
            return
        frame = hourly_frame(ds, args.province)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out_dir / "hourly.csv", encoding="utf-8-sig")
    daily = daily_frame(frame)
    daily.to_csv(args.out_dir / "daily.csv", encoding="utf-8-sig")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
    daily.energy_mwh.where(daily.complete_day).plot(ax=axes[0])
    axes[0].set_ylabel("Simulated energy (MWh/day)")
    daily[[c for c in daily if "capacity_factor" in c]].plot(ax=axes[1])
    axes[1].set_ylabel("Capacity factor (1)")
    fig.tight_layout()
    fig.savefig(args.out_dir / "daily.png", dpi=180)
    plt.close(fig)
    print(f"Saved hourly.csv, daily.csv and daily.png to {args.out_dir}")


if __name__ == "__main__":
    main()
