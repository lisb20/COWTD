# China Onshore Wind Turbine Dataset (COWTD)

COWTD contains 87,714 onshore wind turbine records with construction-year estimates spanning 2014–2024. This repository provides the dataset and standalone scripts illustrating model training, inference, spatial post-processing, construction-year estimation, and validation.

## Dataset

The dataset is available on [figshare](https://doi.org/10.6084/m9.figshare.31939614). The included [COWTD.csv](COWTD.csv) contains the following columns:

| Column | Description |
| --- | --- |
| `turbine_id` | Unique turbine record identifier. |
| `cluster_id` | Spatial cluster identifier; a cluster does not necessarily represent a verified wind farm. |
| `score` | Detection confidence score. A missing value indicates an unmatched score and should not be treated as zero. |
| `year` | Estimated construction year, represented by first-observed presence in the sampled imagery. |
| `lon` | WGS 84 longitude in degrees. |
| `lat` | WGS 84 latitude in degrees. |
| `province` | Province-level administrative region. |

The observation years are 2014, 2017, 2020, 2021, 2022, 2023, and 2024. The 2014 category denotes presence by the earliest observation. Cumulative snapshots can be formed by selecting records with `year` at or before an observation year; these refer to the turbine cohort retained in the early-2024 reference imagery.

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
| `construction_year.py` | Convert box centres to geographic coordinates, merge reference-year detections, and assign construction-year estimates from historical observations. |
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

The temporal reconstruction uses the following directory layout:

```text
runs/
  predict/
    2014/hebei/row_col.txt
    2017/hebei/row_col.txt
    ...
    2024/hebei/row_col.txt
  clusters/
    hebei/clusters.json
```

Tile filenames use `row_col`. Each label line contains normalized YOLO coordinates in the form `class xc yc width height [confidence]`. DBSCAN defaults to `eps=20` and `min_samples=10`, measured in zoom-17 tile coordinates and tile counts, respectively. Empty labels are excluded from clustering. The temporal script writes a new inventory with run-specific identifiers and leaves `score` unassigned.

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
