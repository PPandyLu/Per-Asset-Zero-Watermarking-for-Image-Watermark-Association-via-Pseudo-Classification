# Controlled reconstruction experiments

Each condition directory contains the task design, per-task measurements and available aggregate tables. The controlled cohort comprises 100 registered assets. Open the [45-setting table](direct_metric_summary/direct_metric_response_by_level.csv) for scale, salt-and-pepper noise, Gaussian noise, speckle noise, filtering and compression. For cropping, compare [perceptual match rates](crop_spatial/results/crop_response_by_retain_percent.csv) with [centered-NCC screening rates](crop_spatial/results/cncc_screening_response_by_retain_percent.csv).

To regenerate the corresponding summaries from the repository root, run the commands below with Python 3. Each script writes the summary tables under this directory.

```sh
python experiments/controlled/scripts/summarize_direct_metric_suite.py
python experiments/controlled/scripts/summarize_cncc_screening_responses.py
```
