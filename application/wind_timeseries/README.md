# Wind power time-series usage

`analyze.py` and `example.ipynb` reorganize the reading, aggregation and plotting
examples in `step4_result_chek.ipynb`. The CLI requires `--input`; the notebook
has an editable external `INPUT` path. No source data is bundled.

```bash
python application/wind_timeseries/analyze.py --input /path/to/wind_aggregated_by_province.nc --inspect
python application/wind_timeseries/analyze.py --input /path/to/wind_aggregated_by_province.nc --province 云南省 --out-dir outputs/yunnan
python application/wind_timeseries/analyze.py --input /path/to/wind_sim_ROW_COL_PROVINCE.nc --out-dir outputs/grid
```

## Expected variables

| Scope | Variable | Meaning / unit |
| --- | --- | --- |
| Province | `total_generation(time, province)` | Simulated power, kW |
| Province | `mean_capacity_factor` | Arithmetic grid mean, dimensionless |
| Province | `std_capacity_factor` | Spatial CF standard deviation, dimensionless |
| Province | `weight_capacity_factor` | Capacity-weighted grid mean, dimensionless |
| Province | `total_capacity(province)` | Installed capacity, kW |
| Province | `station_count(province)` | Number of aggregated grid results |
| Grid | `capacity_factor(time[, location])` | CF, dimensionless |
| Grid | `capacity([location])` | Capacity, kW according to source preprocessing |

One location per grid file is expected. If grid capacity has no units attribute,
the source preprocessing convention (kW) is used; other input products must be
checked before reuse. Grid power is calculated as CF × capacity.

Outputs in `--out-dir`: `hourly.csv`, `daily.csv`, `daily.png`. Use separate output
directories for different selections; repeated runs overwrite these outputs.

The source notebook shows 8,784 time samples and 32 province coordinates, starting
at `2023-12-31 23:30`. The script checks regular hourly spacing and keeps source
timestamps. It assigns each one-hour sample to the date of its timestamp:
`daily energy [MWh] = sum(hourly power [kW]) / 1000`. This is an explicit binning
convention, not a verified civil-time interval interpretation. `valid_power_hours`
and `complete_day` identify incomplete days; these days remain in CSV and are
omitted from the daily-energy plot. CF columns use daily arithmetic time means.
An all-missing energy day stays missing rather than becoming zero.

No simulation or province reaggregation is performed here. The original
simulation preparation/run/aggregation scripts remain in the local source
archive for reference; their machine-specific workflows are not release entry points.
