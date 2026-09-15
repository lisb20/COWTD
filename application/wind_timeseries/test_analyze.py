"""Synthetic checks; no external files or downloaded data required."""
import unittest

import numpy as np
import pandas as pd
import xarray as xr

from analyze import daily_frame, hourly_frame


class TimeSeriesTests(unittest.TestCase):
    def grid(self, times):
        return xr.Dataset({
            "capacity_factor": (("time", "location"), np.full((len(times), 1), 0.5)),
            "capacity": ("location", [2000.0]),
        }, coords={"time": times, "location": [0]})

    def test_power_energy_and_boundary_days(self):
        times = pd.date_range("2023-12-31 23:30", periods=48, freq="h")
        frame = hourly_frame(self.grid(times))
        self.assertTrue((frame.power_kw == 1000).all())
        daily = daily_frame(frame)
        self.assertEqual(daily.energy_mwh.tolist(), [1.0, 24.0, 23.0])
        self.assertEqual(daily.complete_day.tolist(), [False, True, False])

    def test_missing_day_is_not_zero(self):
        times = pd.date_range("2024-01-01", periods=48, freq="h")
        frame = hourly_frame(self.grid(times))
        frame.loc["2024-01-01", "power_kw"] = np.nan
        daily = daily_frame(frame)
        self.assertTrue(np.isnan(daily.energy_mwh.iloc[0]))
        self.assertEqual(daily.valid_power_hours.iloc[0], 0)

    def test_reject_irregular_time(self):
        times = pd.to_datetime(["2024-01-01", "2024-01-01 02:00"], format="mixed")
        with self.assertRaisesRegex(ValueError, "regular hourly"):
            hourly_frame(self.grid(times))

    def test_province_selection_and_units(self):
        times = pd.date_range("2024-01-01", periods=24, freq="h")
        ds = xr.Dataset({
            "total_generation": (("time", "province"), np.full((24, 1), 2000.0), {"units": "kW"}),
            **{name: (("time", "province"), np.full((24, 1), 0.5)) for name in
               ["mean_capacity_factor", "std_capacity_factor", "weight_capacity_factor"]},
        }, coords={"time": times, "province": ["test"]})
        self.assertEqual(daily_frame(hourly_frame(ds, "test")).energy_mwh.iloc[0], 48)
        with self.assertRaisesRegex(ValueError, "Select a province"):
            hourly_frame(ds)
        ds.total_generation.attrs["units"] = "MW"
        with self.assertRaisesRegex(ValueError, "units kW"):
            hourly_frame(ds, "test")


if __name__ == "__main__":
    unittest.main()
