# Manual validation samples

These tables contain the authors' manual review results. Each CSV has exactly three columns:

| Column | Description |
| --- | --- |
| `id` | Review sample identifier, not the released inventory's `turbine_id`. |
| `bbox` | Geographic extent of the entire reviewed image, encoded as a JSON array `[west, south, east, north]` in WGS 84 (EPSG:4326), in degrees. This is not an object bounding box. |
| `result` | Spatial reviews: `1` = turbine confirmed, `0` = not confirmed as a turbine. Temporal review: `1` = assigned year agrees with manual review, `0` = disagrees. |

## Experiments

“Ours-only” denotes COWTD records unmatched to the named inventory under the manuscript's spatial matching criterion. “External-only” denotes unmatched reference-inventory records. Confirmation labels refer to the target location in the reviewed imagery.

| File | Reviewed sample | Records | `result=1` | `result=0` |
| --- | --- | ---: | ---: | ---: |
| [national500.csv](national500.csv) | Random sample from the complete COWTD inventory | 500 | 487 | 13 |
| [GRW_2024q2_ours_only200.csv](GRW_2024q2_ours_only200.csv) | COWTD-only relative to GRW 2024Q2 | 200 | 194 | 6 |
| [GRW_2024q2_external_only200.csv](GRW_2024q2_external_only200.csv) | GRW 2024Q2-only relative to COWTD | 200 | 36 | 164 |
| [GonshoreWT2024_ours_only200.csv](GonshoreWT2024_ours_only200.csv) | COWTD-only relative to GonshoreWT2024 | 200 | 172 | 28 |
| [GOWIRES_2025collection_ours_only200.csv](GOWIRES_2025collection_ours_only200.csv) | COWTD-only relative to GOWIRES | 200 | 177 | 23 |
| [cluster_retained200.csv](cluster_retained200.csv) | Retained remote-sensing inventory after clustering and detection rechecking | 200 | 192 | 8 |
| [cluster_removed200.csv](cluster_removed200.csv) | Nationwide sample of candidate tiles removed by clustering | 200 | 14 | 186 |
| [temporal1000.csv](temporal1000.csv) | Manual assessment of assigned years from historical imagery | 1,000 | 867 | 133 |

In every spatial-review table, `result=1` means a turbine is confirmed, not absent. This also applies to groups with few positive labels: `GRW_2024q2_external_only200` contains 36 confirmed turbines (18.0%), while `cluster_removed200` contains 14 confirmed turbines (7.0%) and 186 non-turbine targets (93.0%). The noise fraction in the removed group is therefore calculated from `result=0`.

## Images and identifiers

The inventory-review tables retain the original review identifiers. Their bounding boxes describe the actual 256×256 target-centred review crops from World Imagery Wayback's 8 February 2024 archive (37965), at zoom 17. Such crops need not align with native tile boundaries.

The removed-group identifiers are `N-001`–`N-200`; their bounding boxes describe complete native zoom-17 tiles from the original 2024 image collection. The retained-group table keeps its original inventory-review identifiers and target-centred image extents.

Temporal identifiers are `T001`–`T1000`, corresponding to the original review order. Each row represents one reviewed year assignment, using the same native tile extent across the seven observation years: 2014, 2017, 2020, 2021, 2022, 2023, and 2024. Some historical images are unavailable. These binary labels do not provide per-year presence annotations or corrected construction years. The 1,000 records were sampled across assigned-year and geographic-region strata; their 86.7% agreement is the unweighted sample result.

Only sample identifiers, image extents, and review outcomes are provided here. Satellite images and machine-specific file paths are not included. See [wayback_snapshots.csv](../wayback_snapshots.csv) for historical archive information.
