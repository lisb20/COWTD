# Wind drought frequency and duration

Adapted from the spark01 `release/fig1_wdf_wdd/compute_and_plot.py`.
No hourly extracts or annual caches are bundled.

## Definitions retained from the source

- WDF: yearly hours with CF < 0.1, summarized as mean ± one spatial standard
  deviation and sum across grids active in that year (`capacity_YYYY > 0`).
- WDD: consecutive hours with CF below the grid's baseline P15/P20/P30 threshold.
  Gaps of at most 6 hours are merged when their CF stays at or below baseline P40.
  The yearly statistic averages duration/event-count across active grids with
  at least one event. The sum of WDF across grids depends on the number of grids;
  it is not a national outage duration.
- Default baseline and target years: 2014, 2017, 2020, 2021, 2022, 2023, 2024.
- The original code normalizes 8,784-hour input years to 8,760 hours by removing
  positions 1416–1439. This assumes a complete hourly calendar starting January 1;
  verify that assumption for any new source. Event statistics are computed per year.

## Inputs and commands

Recompute requires `{YYYY}.nc` with `wcf(time, lat, lon)` and a CSV with `lon`,
`lat`, and `capacity_YYYY`. The source used wind capacity table v3. Defaults
restrict the coordinate extent to 70–140° E, 15–55° N; this rectangular filter
alone is not an administrative boundary mask.

```bash
python application/wind_drought/compute_and_plot.py --compute --plot --data-dir /path/to/annual_wcf --meta-wind-csv /path/to/grid_capacity.csv --cache-dir /path/to/new_cache --out-dir outputs/drought
python application/wind_drought/compute_and_plot.py --plot --cache-dir /path/to/yearly_cache --out-dir outputs/drought
```

Plot-only expects `wind_yearly_totals_v4_2014_2024*.npz`, with the schema produced
by this script. `--plot-annot` adds annotated figure variants. The full dependency
list is required because the original module imports pandas and xarray at startup.

Use a **new cache directory whenever inputs, grid ordering, years or thresholds
change**. The inherited cache implementation uses filenames and point counts,
not content hashes; matching point counts do not prove matching locations.
Legacy cache reuse is disabled by default. `--legacy-cache-dirs` enables explicit
reuse when the caller has verified identity. Plot-only reads stored statistics;
changing compute options does not recalculate them. Keep one intended annual
aggregate per cache directory because plotting selects the newest matching file.

`NATURE_ENERGY_ROOT` may point to an external root containing the original `data/`
layout; explicit input options are clearer for a relocated release.
