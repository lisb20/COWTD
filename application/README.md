# Application and usage notes

Code-only examples for applications of the China Onshore Wind Turbine Dataset
(COWTD). No NetCDF, raster, capacity table, statistical cache, or generated figure
is bundled. Provide inputs separately using the command-line options below.

## Modules

| Directory | Purpose | External inputs |
| --- | --- | --- |
| [wind_timeseries](wind_timeseries/README.md) | Inspect grid/province CF and simulated power; export hourly/daily tables and plots | RESKit grid result or province-aggregated NetCDF |
| [wind_drought](wind_drought/README.md) | Compute/plot wind drought frequency and duration (WDF/WDD) | Annual gridded `wcf` NetCDF and capacity table, or annual statistics cache |
| [suitable_area_share](suitable_area_share/README.md) | Compute/plot suitability share of occupied wind/solar grid cells | Capacity tables and suitability rasters, or statistics CSVs |

The two NetCDF families above are **different input formats**. The downloaded
RESKit grid/province outputs are not direct inputs to the WDF/WDD pipeline.
The suitability pipeline retains the original wind and solar comparison; its
current compute/plot workflow requires both input sets.

## Setup and quick start

Python 3.10 or later. Run from the repository root:

```bash
python -m pip install -r application/requirements.txt
python application/wind_timeseries/analyze.py --input /path/to/wind_aggregated_by_province.nc --inspect
python application/wind_timeseries/analyze.py --input /path/to/wind_aggregated_by_province.nc --province 云南省 --out-dir outputs/yunnan
python application/wind_drought/compute_and_plot.py --plot --cache-dir /path/to/yearly_cache --out-dir outputs/drought
python application/suitable_area_share/compute_and_plot.py --plot --cache-dir /path/to/stats_cache --out-dir outputs/suitability
```

Install `application/requirements-recompute.txt` for suitability raster computations. A
Jupyter installation is needed only for `wind_timeseries/example.ipynb`.
Quote paths containing spaces. Every command's `--help` works without data.

## Interpretation

- These wind power series are simulation outputs. They are not measurements of
  delivered electricity, curtailment, or dispatch.
- A grid-level aggregate is not necessarily a verified individual wind farm.
- Capacity-factor arithmetic means and capacity-weighted means differ.
- `total_generation` in the source province file is **power in kW** despite its
  name. Hourly power must be integrated over time to obtain energy.
- Dates are kept as encoded. No timezone or half-hour timestamp correction is
  inferred. Incomplete boundary days are flagged by the time-series tool.

See [provenance and validation](docs/PROVENANCE.md) for source mapping and limits.
Use the dataset/manuscript citation in the repository's main README. This
directory does not assign a new software or third-party data licence.
