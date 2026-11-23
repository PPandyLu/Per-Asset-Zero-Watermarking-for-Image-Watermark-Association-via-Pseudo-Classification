# Per-Asset Zero-Watermarking for Image–Watermark Association via Pseudo-Classification

Research data and analysis code accompanying the article by Junyan Lu, Guoqing Li, Long Zhao and Lianchong Zhang. The experiments measure whether a registered key reconstructs a reference watermark from a candidate remote sensing image.

## Explore the results

| Question | Start here |
|---|---|
| How often did the perceptual assessment return a match? | [Decision rates and confidence intervals](figures/stratified_statistics/overall_decision_rates.csv) · [Positive and negative task records](experiments/batch/) |
| How well did an objective metric separate the two cohorts? | [ROC curves and AUCs](figures/roc/) · [Ten calibration/evaluation splits](experiments/screening/) |
| What happened as cropping, noise, filtering or compression increased? | [Controlled measurements and summary scripts](experiments/controlled/) |
| Where did the model and human panel disagree? | [Collective blind-review labels and images](experiments/blind_review/) |
| Did the integrity checks behave as expected? | [140 protocol checks](experiments/integrity/protocol_check_results.csv) |
| What did preprocessing cost at each image size? | [Benchmark setup and timings](experiments/runtime/) |
| How do I register a key and run an extraction? | [Method code and input layout](method/) |

The main perceptual assessment returned 4,573 matches among 5,000 positive tasks and 163 matches among 3,000 negative tasks. These decisions come from the frozen model and prompt. The objective screen uses centered NCC. The human assessment records a separate three-level collective judgement.

**Browse by paper figure:** [Figures 22–28, with plots and source tables](figures/README.md). Figures are numbered as in the final manuscript.

## Working with the files

CSV and JSON files are UTF-8. Open the CSV summaries directly on GitHub or download the repository for analysis. The positive and negative experiment folders each contain a judgement table and two ZIP archives. Extract each archive inside its own experiment folder so the relative paths in the CSV resolve correctly.

The repository distributes derived measurements, watermark comparison panels, and selected research scripts. For experiments on the original xBD imagery, obtain the imagery separately from its distributor. Use [SHA256SUMS](SHA256SUMS) to verify downloaded files after extracting the repository and before unpacking the nested archives.
