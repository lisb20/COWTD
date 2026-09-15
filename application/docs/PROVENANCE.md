# Provenance and validation

Prepared on 2026-09-15 as a code-only application supplement.

| Supplied source | Release counterpart |
| --- | --- |
| LLM1 RESKit `examples/3_wind/step4_result_chek.ipynb` | `wind_timeseries/analyze.py`, `wind_timeseries/example.ipynb` |
| spark01 `release/fig1_wdf_wdd/compute_and_plot.py` | `wind_drought/compute_and_plot.py` |
| spark01 `release/suitable_area_share/compute_and_plot.py` | `suitable_area_share/compute_and_plot.py` |

Reference sources were collected in a local `_source_downloads/` directory,
which is not included in the public package. The source notebook's execution outputs were
cleared. Data files and statistical caches downloaded during initial collection
were removed when the scope was clarified to code only. This restriction applies
to the application directory; the updated 120,849-record COWTD inventory is
provided separately at the repository root.

Changes: parameterized time-series input, explicit power/energy units and hourly
validation, missing/partial-day handling, a clean notebook, portable raw-input
roots, functional suitability `--base-dir`, disabled implicit legacy-cache lookup,
and code-only documentation. WDF/WDD and suitability metric algorithms otherwise
retain the source implementation.

Unit evidence: source `step1_process_plant.py` converts MW to kW; source
`step3_output_process.py` computes CF × capacity and assigns `total_generation`
units `kW`. The original notebook used inconsistent MWh labels, corrected here.

Validation scope: command-line imports/help, Python/notebook syntax, and synthetic
in-memory checks of power conversion, daily energy, missing values and invalid
time spacing. No full scientific recomputation with external NetCDF/GeoTIFF input
is claimed. Existing scientific assumptions and cache-identity limitations are
documented in the module READMEs.
