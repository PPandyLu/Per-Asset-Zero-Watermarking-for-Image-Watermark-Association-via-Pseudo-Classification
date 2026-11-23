# Registration and extraction

The scripts implement the batch experiment on per-asset random forests and circularly masked Hu-moment descriptors. Use Python 3 with NumPy, OpenCV (including JPEG 2000 support), scikit-learn, scikit-image and joblib.

Prepare a working directory with one folder per asset under `samples/`. In each asset folder, provide `original.png` and a 70 × 70 binary `reference_watermark.png`. The following commands register one key per asset, generate the fixed-seed compound attacks and extract the watermarks. Replace `WORKDIR` with your working directory. Outputs are written there.

```sh
python method/train_clean_batch.py --experiment WORKDIR
python method/generate_random_attack_chains.py --experiment WORKDIR
python method/extract_attack_metrics.py --experiment WORKDIR
```

The registration script resizes the input to 100 × 100 pixels. The extraction script reads the locally generated model and attack manifest. For the paper's measurements and original cohort identifiers, see the [batch evaluation](../experiments/batch/).
