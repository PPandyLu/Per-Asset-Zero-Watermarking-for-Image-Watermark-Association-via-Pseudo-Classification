"""Train one Hu-moment random-forest zero-watermark key per selected xBD image."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier


IMAGE_SIZE = 100
WATERMARK_SIZE = 70
WINDOW_SIZE = IMAGE_SIZE - WATERMARK_SIZE + 1
N_ESTIMATORS = 100
RANDOM_STATE = 42


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_image(path: Path, flags: int) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), flags)
    if image is None:
        raise RuntimeError(f"Could not decode {path}")
    return image


def write_image(path: Path, image: np.ndarray) -> None:
    success, encoded = cv2.imencode(path.suffix, image)
    if not success:
        raise RuntimeError(f"Could not encode {path}")
    encoded.tofile(path)


def circle_mask(size: int) -> np.ndarray:
    center = size // 2
    y, x = np.ogrid[:size, :size]
    return (x - center) ** 2 + (y - center) ** 2 <= center**2


def signed_log_hu(patch: np.ndarray, mask: np.ndarray) -> np.ndarray:
    circular_patch = np.zeros_like(patch)
    circular_patch[mask] = patch[mask]
    hu = cv2.HuMoments(cv2.moments(circular_patch)).flatten()
    return -np.sign(hu) * np.log10(np.abs(hu) + 1e-10)


def feature_matrix(image: np.ndarray) -> np.ndarray:
    if image.shape != (IMAGE_SIZE, IMAGE_SIZE):
        raise ValueError(f"Expected {IMAGE_SIZE}x{IMAGE_SIZE}, got {image.shape}")
    mask = circle_mask(WINDOW_SIZE)
    features = np.empty((WATERMARK_SIZE * WATERMARK_SIZE, 7), dtype=np.float64)
    index = 0
    for row in range(WATERMARK_SIZE):
        for column in range(WATERMARK_SIZE):
            features[index] = signed_log_hu(
                image[row : row + WINDOW_SIZE, column : column + WINDOW_SIZE], mask
            )
            index += 1
    return features


def binary_metrics(reference: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    ref = reference > 127
    pred = prediction > 127
    true_positive = int(np.logical_and(ref, pred).sum())
    false_positive = int(np.logical_and(~ref, pred).sum())
    false_negative = int(np.logical_and(ref, ~pred).sum())
    bit_error_rate = float(np.not_equal(ref, pred).mean())
    intersection = int(np.logical_and(ref, pred).sum())
    union = int(np.logical_or(ref, pred).sum())
    iou = float(intersection / union) if union else 1.0
    ref_float = reference.astype(np.float64).ravel()
    pred_float = prediction.astype(np.float64).ravel()
    denominator = np.linalg.norm(ref_float) * np.linalg.norm(pred_float)
    ncc = float(np.dot(ref_float, pred_float) / denominator) if denominator else 0.0
    return {
        "ncc": ncc,
        "bit_error_rate": bit_error_rate,
        "iou": iou,
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
    }


def process_sample(sample_dir: Path, workers: int) -> dict[str, object]:
    original = read_image(sample_dir / "original.png", cv2.IMREAD_GRAYSCALE)
    original = cv2.resize(original, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)
    reference = read_image(sample_dir / "reference_watermark.png", cv2.IMREAD_GRAYSCALE)
    if reference.shape != (WATERMARK_SIZE, WATERMARK_SIZE):
        raise ValueError(f"Unexpected reference watermark shape: {reference.shape}")

    feature_start = time.perf_counter()
    features = feature_matrix(original)
    feature_seconds = time.perf_counter() - feature_start

    training_start = time.perf_counter()
    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        random_state=RANDOM_STATE,
        class_weight="balanced",
        n_jobs=workers,
    )
    model.fit(features, reference.ravel())
    training_seconds = time.perf_counter() - training_start

    model_path = sample_dir / "model.joblib"
    joblib.dump(model, model_path)
    prediction_start = time.perf_counter()
    prediction = model.predict(features).reshape((WATERMARK_SIZE, WATERMARK_SIZE)).astype(np.uint8)
    prediction_seconds = time.perf_counter() - prediction_start

    extracted_dir = sample_dir / "extracted"
    metrics_dir = sample_dir / "metrics"
    extracted_dir.mkdir(exist_ok=True)
    metrics_dir.mkdir(exist_ok=True)
    extracted_path = extracted_dir / "clean.png"
    write_image(extracted_path, prediction)
    metrics = binary_metrics(reference, prediction)
    result = {
        "asset_id": sample_dir.name,
        "image_size": IMAGE_SIZE,
        "watermark_size": WATERMARK_SIZE,
        "window_size": WINDOW_SIZE,
        "feature_dimension": 7,
        "feature_samples": int(features.shape[0]),
        "random_forest": {
            "n_estimators": N_ESTIMATORS,
            "random_state": RANDOM_STATE,
            "class_weight": "balanced",
            "max_depth": None,
            "n_jobs": workers,
        },
        "feature_seconds": round(feature_seconds, 6),
        "training_seconds": round(training_seconds, 6),
        "prediction_seconds": round(prediction_seconds, 6),
        "model_bytes": model_path.stat().st_size,
        "model_sha256": sha256(model_path),
        "extracted_sha256": sha256(extracted_path),
        "metrics": metrics,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (metrics_dir / "clean_result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=-1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    experiment = args.experiment.resolve()
    sample_dirs = sorted(path for path in (experiment / "samples").iterdir() if path.is_dir())
    if args.limit is not None:
        sample_dirs = sample_dirs[: args.limit]
    results: list[dict[str, object]] = []
    for index, sample_dir in enumerate(sample_dirs, start=1):
        result_path = sample_dir / "metrics" / "clean_result.json"
        if result_path.exists() and not args.force:
            result = json.loads(result_path.read_text(encoding="utf-8"))
            print(f"[{index}/{len(sample_dirs)}] {sample_dir.name}: already complete")
        else:
            result = process_sample(sample_dir, args.workers)
            print(
                f"[{index}/{len(sample_dirs)}] {sample_dir.name}: "
                f"train={result['training_seconds']:.2f}s, BER={result['metrics']['bit_error_rate']:.6f}"
            )
        results.append(result)

    result_dir = experiment / "results"
    result_dir.mkdir(exist_ok=True)
    with (result_dir / "clean_summary.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "asset_id", "feature_seconds", "training_seconds", "prediction_seconds",
                "model_bytes", "ncc", "bit_error_rate", "iou",
            ],
        )
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "asset_id": result["asset_id"],
                    "feature_seconds": result["feature_seconds"],
                    "training_seconds": result["training_seconds"],
                    "prediction_seconds": result["prediction_seconds"],
                    "model_bytes": result["model_bytes"],
                    "ncc": result["metrics"]["ncc"],
                    "bit_error_rate": result["metrics"]["bit_error_rate"],
                    "iou": result["metrics"]["iou"],
                }
            )


if __name__ == "__main__":
    main()
