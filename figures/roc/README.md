# ROC curves for manuscript Figure 22

The primary split contains 6,344 held-out tasks. Use `roc_points.csv` to plot true-positive rate against false-positive rate in row order for each metric. The summary file lists the AUC for each curve. The held-out score table includes the event, registered asset group and ground-truth label (1 for positive controls). Several task rows per registered asset are expected.

The split seed is `objective_rule_repeat_0`. Within each event, order distinct asset IDs by the SHA-256 digest of `seed|event|asset` and reserve the first 20% (rounded up) for calibration. Rank centered NCC, masked SSIM and IoU from highest to lowest and BER from lowest to highest. Advance the curve once for each group of tied scores. The curves integrate to the four published single-split AUCs.

The working figure identifier in the experimental archive was Fig07. The typeset manuscript labels this plot Figure 22.
