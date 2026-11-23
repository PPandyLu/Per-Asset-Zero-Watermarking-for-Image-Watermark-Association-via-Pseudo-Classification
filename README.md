# Per-Asset Zero-Watermarking for Image–Watermark Association via Pseudo-Classification

This data collection covers the 5,000-task positive-control perceptual assessment and the four objective-metric ROC curves in Figure 7. The ROC analysis includes both positive and negative examples in the primary held-out split.

## Positive-control assessment

Open `positive_control/judgements.csv` for one row per task. There are 200 registered assets and 5,000 tasks. The frozen assessment classified 4,573 tasks as matches and 427 as non-matches (91.46% match rate). The `match` column is the model decision. The `confidence` and `match_score` fields describe the model response.

Extract both ZIP files **inside `positive_control/`**. Each archive already contains its top-level directory. The two relative paths in each CSV row then open the corresponding PNG comparison panel and JSON record. The filenames use the same task ID. Each panel pairs a reference watermark on the left with the reconstructed watermark on the right.

| File | Contents |
|---|---|
| `judgements.csv` | Task and asset IDs, attack labels, model decision, model scores, explanation, timing and relative file paths. |
| `panels.zip` | 5,000 original PNG panels in `panels/`. |
| `responses.zip` | 5,000 JSON records in `responses/`. Each contains the manifest row, model response and inference settings. |
| `prompt.txt` | Assessment instructions transcribed from Supplementary Material S1. |
| `request_profile.json` | Inference settings common to all 5,000 tasks. |

The JSON files are publication copies. Provider request IDs, the API-key environment variable name, the service endpoint and the run directory were removed. Panel paths are relative to this collection. Scientific response content, scores and timestamps are retained. The PNG files are byte-identical to the archived experiment panels.

## Figure 7

| File in `figure_7/` | Contents |
|---|---|
| `auc_summary.csv` | Original four-row figure summary, with held-out sample size and AUC for each metric. |
| `held_out_scores.csv` | The 6,344 held-out task rows, including asset group, event, binary ground truth and the four metric values. Multiple rows per asset are expected. |
| `roc_points.csv` | False-positive and true-positive rates in curve order, grouped by metric. Plot the two columns against each other to draw each ROC curve. |

The primary split uses seed `objective_rule_repeat_0`. Within each event, distinct asset IDs are ordered by the SHA-256 digest of `seed|event|asset`. The first 20% (rounded up) are reserved for calibration and the remaining assets are held out. Label 1 denotes a positive task. Rank centered NCC, masked SSIM and IoU from highest to lowest, and BER from lowest to highest. Advance the curve once for each tied-score group. Trapezoidal integration reproduces all four AUCs in the original figure summary to within 1e-12. These are single-split results; the paper also reports a median across ten splits.

## File integrity

`SHA256SUMS` contains SHA-256 digests for the data files and this README. The CSV files use UTF-8. No credentials are needed to inspect the archived data.
