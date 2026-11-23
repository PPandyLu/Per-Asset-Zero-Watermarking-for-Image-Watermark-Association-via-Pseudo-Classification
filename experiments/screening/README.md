# Objective screening

The [published split results](results/repeat_split_results.csv) contain ten event-stratified calibration/evaluation partitions. Rows for one registered source image stay in the same partition. The [threshold-stability table](results/centered_ncc_fixed_threshold_stability.csv) evaluates the rounded 0.160 centered-NCC setting. [Figure 22](../../figures/roc/) uses the primary held-out split.

From the repository root, run the following with Python, NumPy 2.x and pandas 2.3.3 installed. This version of pandas reproduces the archived score-tie handling when reading decimal values from CSV. The script reads selected columns from the original positive and negative metric files in `inputs/` and writes calculations to `experiments/screening/recomputed/`.

```sh
python experiments/screening/scripts/explore_objective_rules.py
```
