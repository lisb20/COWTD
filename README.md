# China Onshore Wind Turbine Dataset (COWTD)

COWTD contains **120,849 onshore wind turbine records** with first-observed-year labels spanning 2014–2024. The updated inventory retains 87,714 locations from nationwide satellite-image detection and adds 33,135 nonduplicate locations through OSM-guided image verification. This repository provides the inventory, standalone detection and validation examples, and application code for siting suitability, wind power generation, and wind drought exposure.

## Dataset

The dataset is available on [figshare](https://doi.org/10.6084/m9.figshare.31939614). The included [COWTD.csv](COWTD.csv) contains the following columns:

| Column | Description |
| --- | --- |
| `turbine_id` | Unique turbine record identifier. |
| `osm_matched` | `1` if matched to an OSM-derived GOWIRES turbine record in the paper's spatial comparison; `0` otherwise. See the matching definition below. |
| `score` | Object-detection confidence in the reference imagery. Scores are missing for 1,726 records involving multiple overlapping detection boxes; missing values should not be treated as zero. |
| `year` | First-observed year in the sampled satellite imagery. |
| `lon` | WGS 84 longitude in degrees. |
| `lat` | WGS 84 latitude in degrees. |
| `province` | Province-level administrative region. |

The observation years are 2014, 2017, 2020, 2021, 2022, 2023, and 2024. The 2014 category denotes presence by the earliest observation. Cumulative snapshots can be formed by selecting records with `year` at or before an observation year; these refer to the turbine cohort retained in the early-2024 reference imagery.

The observation intervals are uneven: three years through 2020 and one year thereafter. Image-acquisition timing and identification errors introduce temporal uncertainty. Compare changes across the available observation intervals; snapshot dates are not exact construction dates. Turbines removed before the early-2024 reference imagery are not included. Higher `score` thresholds select detections with greater model confidence while reducing coverage.

`osm_matched` preserves the paper's maximum-cardinality, one-to-one matching within 100 m against [OSM label](https://doi.org/10.5281/zenodo.18768952), using the common comparison boundary. It contains 104,836 matches and 16,013 zeros.

| Observation year | Cumulative turbine count |
| --- | ---: |
| 2014 | 9,675 |
| 2017 | 31,777 |
| 2020 | 49,378 |
| 2021 | 77,435 |
| 2022 | 95,468 |
| 2023 | 107,122 |
| 2024 | 120,849 |

The public CSV uses `osm_matched` for the source export's `OSM-recorded` field; the values and all turbine records are preserved.

To load and select an observation-year inventory:

```python
import pandas as pd

turbines = pd.read_csv("COWTD.csv")
inventory_2020 = turbines.loc[turbines["year"] <= 2020]
provincial_counts = inventory_2020.groupby("province").size()
```

Coordinates are in WGS 84 (EPSG:4326). Transform them to a suitable projected coordinate system when calculating planar distances or areas.

## Manual validation samples

[manual_annotations](manual_annotations/README.md) provides one CSV per review experiment, containing only sample identifiers, geographic image bounding boxes, and manual review outcomes. It includes national inventory validation, unmatched-inventory comparisons, clustering-removed candidates, and temporal validation. The folder documents label definitions and sample overlap; satellite images can be accessed through World Imagery Wayback API.

## Requirements

Use Python 3.10 or later and install the dependencies:

```bash
pip install -r requirements.txt
```

Run the scripts from this directory. Set the input paths, output paths, and device options at the top of each script before running. Source imagery, training annotations, and model weights are supplied separately by the user.

## Scripts

| File | Purpose |
| --- | --- |
| `train.py` | Train a YOLO detector using local annotated imagery. |
| `predict.py` | Run inference on local image tiles and save bounding boxes with confidence scores. |
| `post_process.py` | Apply DBSCAN to nonempty detection tiles within each province and export retained tile groups. |
| `construction_year.py` | Convert box centres to geographic coordinates, merge reference-year detections, and identify first-observed years from historical observations. |
| `validation.py` | Evaluate the detector, summarize turbine counts, calculate count–capacity correlations, compare inventories, and summarize temporal review results. |
| `data.example.yaml` | Example configuration for the training, validation, and test datasets. |
| `wayback_snapshots.csv` | World Imagery Wayback release IDs, archive dates, source URLs, and observation-year selection. Archive dates are distinct from local imagery acquisition dates. |

## Example workflow

1. Prepare local imagery and YOLO annotations using `data.example.yaml`. Include positive samples and manually confirmed negative samples in the training, validation, and test sets. Negative images have empty label files.
2. Configure and run `train.py`. The training example uses 300 epochs and a 256-pixel input size.
3. Set `MODE = "detection"` and `SPLIT = "val"` in `validation.py`. Select the confidence threshold from the validation F1-confidence curve, and use the independent test split for evaluation.
4. Set the selected threshold in `predict.py`, then run inference for each province and observation year.
5. Run `post_process.py` on each province's 2024 detections. Configure the province list and input paths in `construction_year.py`, then construct the temporal records.

```bash
python train.py
python validation.py
python predict.py
python post_process.py
python construction_year.py
```

## Applications

See [application/README.md](application/README.md) for installation and command-line examples. Each module documents its external input formats and statistical definitions:

| Application | Code and instructions |
| --- | --- |
| Siting suitability | [suitable_area_share](application/suitable_area_share/README.md): mean suitable-area fraction across occupied grid cells. |
| Wind power generation | [wind_timeseries](application/wind_timeseries/README.md): inspect simulated power and capacity factors, and integrate hourly power to daily energy. |
| Wind drought exposure | [wind_drought](application/wind_drought/README.md): frequency and duration statistics across active grid cells. |

Application dependencies are listed separately in `application/requirements.txt`; raster recomputation also requires `application/requirements-recompute.txt`. Supply meteorological inputs, capacity tables, and suitability rasters separately. These external data and generated outputs are not bundled in the repository.

## Citation

If you use COWTD or these scripts, please cite the accompanying manuscript:

Ding, J., Li, S., Sun, C., Zhang, J., Qi, J., Zhang, R., Huang, R., Li, Z., Luo, B., Huang, H., Yan, X., Xu, G., Yuan, Y., Luo, Y., and Li, Y. *A temporal dataset of onshore wind turbines in China derived from satellite imagery.* Manuscript.

```bibtex
@unpublished{ding_cowtd,
  author = {Ding, Jingtao and Li, Sibo and Sun, Chongjing and Zhang, Jiadong
            and Qi, Jiacheng and Zhang, Ruoyang and Huang, Renjun and Li, Zhaoyi
            and Luo, Baozhen and Huang, Hao and Yan, Xiaohui and Xu, Guangkuo
            and Yuan, Yuan and Luo, Yong and Li, Yong},
  title = {A temporal dataset of onshore wind turbines in China derived from satellite imagery},
  note = {Manuscript}
}
```

Dataset citation: Ding, J. and Li, S. (2026). *A temporal dataset of onshore wind turbines in China derived from satellite imagery.* figshare. https://doi.org/10.6084/m9.figshare.31939614
