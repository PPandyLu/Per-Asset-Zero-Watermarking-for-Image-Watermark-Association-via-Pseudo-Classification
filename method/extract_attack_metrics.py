"""Extract watermarks from all attacks and compute reproducible quantitative metrics."""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import time
from pathlib import Path

import cv2
import joblib
import numpy as np
from skimage.metrics import structural_similarity

from train_clean_batch import IMAGE_SIZE, binary_metrics, read_image, write_image


def normalized_cross_correlation(first: np.ndarray, second: np.ndarray) -> float:
    first = first.astype(np.float64).ravel()
    second = second.astype(np.float64).ravel()
    denominator = np.linalg.norm(first) * np.linalg.norm(second)
    return float(np.dot(first, second) / denominator) if denominator else 0.0


def centered_normalized_cross_correlation(first: np.ndarray, second: np.ndarray) -> float:
    first = first.astype(np.float64).ravel()
    second = second.astype(np.float64).ravel()
    first -= first.mean()
    second -= second.mean()
    denominator = np.linalg.norm(first) * np.linalg.norm(second)
    return float(np.dot(first, second) / denominator) if denominator else 0.0


def safe_ssim(first: np.ndarray, second: np.ndarray) -> float:
    side = min(first.shape[:2])
    if side < 3:
        return float("nan")
    win_size = min(7, side if side % 2 == 1 else side - 1)
    return float(structural_similarity(first, second, data_range=255, win_size=win_size))


def best_overlap(reference: np.ndarray, extracted: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, int | float]]:
    """Align by selecting the overlap with maximal normalized cross-correlation."""
    reference = reference.astype(np.uint8)
    extracted = extracted.astype(np.uint8)
    if extracted.shape[0] >= reference.shape[0] and extracted.shape[1] >= reference.shape[1]:
        response = cv2.matchTemplate(extracted, reference, cv2.TM_CCORR_NORMED)
        _, _, _, maximum = cv2.minMaxLoc(response)
        x, y = maximum
        aligned_extracted = extracted[y : y + reference.shape[0], x : x + reference.shape[1]]
        aligned_reference = reference
        reference_offset_x, reference_offset_y = 0, 0
        extracted_offset_x, extracted_offset_y = x, y
    else:
        response = cv2.matchTemplate(reference, extracted, cv2.TM_CCORR_NORMED)
        _, _, _, maximum = cv2.minMaxLoc(response)
        x, y = maximum
        aligned_reference = reference[y : y + extracted.shape[0], x : x + extracted.shape[1]]
        aligned_extracted = extracted
        reference_offset_x, reference_offset_y = x, y
        extracted_offset_x, extracted_offset_y = 0, 0
    coverage = aligned_reference.size / reference.size
    return aligned_reference, aligned_extracted, {
        "reference_offset_x": reference_offset_x,
        "reference_offset_y": reference_offset_y,
        "extracted_offset_x": extracted_offset_x,
        "extracted_offset_y": extracted_offset_y,
        "overlap_width": aligned_reference.shape[1],
        "overlap_height": aligned_reference.shape[0],
        "coverage": coverage,
    }


def overlap_metrics(reference: np.ndarray, extracted: np.ndarray) -> dict[str, float | int]:
    aligned_reference, aligned_extracted, alignment = best_overlap(reference, extracted)
    values = binary_metrics(aligned_reference, aligned_extracted)
    values["masked_ncc"] = normalized_cross_correlation(aligned_reference, aligned_extracted)
    values["centered_ncc"] = centered_normalized_cross_correlation(aligned_reference, aligned_extracted)
    values["masked_ssim"] = safe_ssim(aligned_reference, aligned_extracted)
    values.update(alignment)
    return values


def rotate_bound(image: np.ndarray, angle: float, border_value: int) -> np.ndarray:
    if abs(angle) < 1e-9:
        return image
    height, width = image.shape[:2]
    center = (width / 2, height / 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos_value = abs(matrix[0, 0])
    sin_value = abs(matrix[0, 1])
    new_width = int(round(height * sin_value + width * cos_value))
    new_height = int(round(height * cos_value + width * sin_value))
    matrix[0, 2] += (new_width - width) / 2
    matrix[1, 2] += (new_height - height) / 2
    return cv2.warpAffine(
        image,
        matrix,
        (new_width, new_height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value,
    )


def normalized_angle(angle: float) -> float:
    normalized = ((angle + 180.0) % 360.0) - 180.0
    if abs(normalized + 180.0) < 1e-9:
        normalized = 180.0
    return round(normalized, 6)


def add_angle(candidates: list[float], angle: float) -> None:
    normalized = normalized_angle(angle)
    if not any(abs(existing - normalized) < 1e-9 for existing in candidates):
        candidates.append(normalized)


def parse_rotation_token(token: str) -> float | None:
    if token.startswith("rotate_"):
        try:
            return float(token.split("_", 1)[1])
        except ValueError:
            return None
    return None


def attack_rotation_angles(attack: dict[str, str]) -> list[float]:
    parameters = json.loads(attack["parameters"])
    angles: list[float] = []
    if attack["attack_family"] == "rotation":
        angles.append(float(parameters["angle"]))
    elif attack["attack_family"] == "compound":
        for operation in parameters.get("operations", []):
            angle = parse_rotation_token(str(operation))
            if angle is not None:
                angles.append(angle)
    elif attack["attack_family"] == "random_chain":
        for operation in parameters.get("operations", []):
            if operation.get("family") == "rotation":
                angles.append(float(operation["parameters"]["angle"]))
    return angles


def rotation_alignment_candidates(attack: dict[str, str]) -> list[float]:
    angles = attack_rotation_angles(attack)
    candidates = [0.0]
    for angle in angles:
        add_angle(candidates, -angle)
        add_angle(candidates, angle)
    if len(angles) > 1:
        total = sum(angles)
        add_angle(candidates, -total)
        add_angle(candidates, total)
    return candidates


def geometry_aligned_metrics(
    reference: np.ndarray, extracted: np.ndarray, attack: dict[str, str]
) -> dict[str, float | int]:
    border_value = 255 if float(reference.mean()) >= 127.5 else 0
    best_values: dict[str, float | int] | None = None
    best_score = -float("inf")
    best_angle = 0.0

    for angle in rotation_alignment_candidates(attack):
        candidate = rotate_bound(extracted, angle, border_value)
        values = overlap_metrics(reference, candidate)
        score = float(values["centered_ncc"])
        if score > best_score:
            best_score = score
            best_values = values
            best_angle = angle

    if best_values is None:
        raise AssertionError("No alignment candidates were evaluated")

    return {
        "geometry_aligned_angle": best_angle,
        "geometry_aligned_ncc": best_values["masked_ncc"],
        "geometry_aligned_centered_ncc": best_values["centered_ncc"],
        "geometry_aligned_ssim": best_values["masked_ssim"],
        "geometry_aligned_bit_error_rate": best_values["bit_error_rate"],
        "geometry_aligned_iou": best_values["iou"],
        "geometry_aligned_coverage": best_values["coverage"],
        "geometry_reference_offset_x": best_values["reference_offset_x"],
        "geometry_reference_offset_y": best_values["reference_offset_y"],
        "geometry_extracted_offset_x": best_values["extracted_offset_x"],
        "geometry_extracted_offset_y": best_values["extracted_offset_y"],
        "geometry_overlap_width": best_values["overlap_width"],
        "geometry_overlap_height": best_values["overlap_height"],
    }


def metrics(reference: np.ndarray, extracted: np.ndarray, attack: dict[str, str]) -> dict[str, float | int]:
    values = overlap_metrics(reference, extracted)
    values.update(geometry_aligned_metrics(reference, extracted, attack))
    return values


def extraction_dimensions(
    original: np.ndarray, attacked: np.ndarray, attack: dict[str, str]
) -> tuple[int, int]:
    """Replicate the original method's physical-resolution-based resampling."""
    original_height, original_width = original.shape[:2]
    attacked_height, attacked_width = attacked.shape[:2]
    scale_correction = 1.0
    parameters = json.loads(attack["parameters"])
    if attack["attack_family"] == "scale":
        scale_correction = 1.0 / float(parameters["factor"])
    elif attack["attack_family"] == "compound" and attack["attack_name"] == "rotate_045__scale_0.50__jpeg_qf_30":
        scale_correction = 2.0
    elif attack["attack_family"] == "random_chain":
        for operation in parameters["operations"]:
            if operation["family"] == "scale":
                scale_correction /= float(operation["parameters"]["factor"])
    height = round(IMAGE_SIZE * attacked_height / original_height * scale_correction)
    width = round(IMAGE_SIZE * attacked_width / original_width * scale_correction)
    if height < 31 or width < 31:
        raise ValueError(f"Attack is too small for the 31x31 extraction window: {width}x{height}")
    return width, height


REQUIRED_RECORD_FIELDS = {
    "geometry_aligned_angle",
    "geometry_aligned_ncc",
    "geometry_aligned_centered_ncc",
    "geometry_aligned_ssim",
    "geometry_aligned_bit_error_rate",
    "geometry_aligned_iou",
    "geometry_aligned_coverage",
}


def process_sample(args: tuple[str, str, list[dict[str, str]], bool]) -> list[dict[str, object]]:
    experiment_text, sample_dir_text, attacks, force = args
    experiment = Path(experiment_text)
    sample_dir = Path(sample_dir_text)
    original = read_image(sample_dir / "original.png", cv2.IMREAD_COLOR)
    reference = read_image(sample_dir / "reference_watermark.png", cv2.IMREAD_GRAYSCALE)
    model = joblib.load(sample_dir / "model.joblib")
    model.n_jobs = 1
    output_dir = sample_dir / "extracted" / "attacks"
    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []

    for attack in attacks:
        attack_path = experiment / attack["relative_output_path"]
        output_path = output_dir / attack["attack_family"] / f"{attack['attack_name']}.png"
        result_path = sample_dir / "metrics" / "attacks" / attack["attack_family"] / f"{attack['attack_name']}.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.parent.mkdir(parents=True, exist_ok=True)
        if result_path.exists() and output_path.exists() and not force:
            record = json.loads(result_path.read_text(encoding="utf-8"))
            if REQUIRED_RECORD_FIELDS.issubset(record):
                records.append(record)
                continue

        attacked = read_image(attack_path, cv2.IMREAD_COLOR)
        resize_width, resize_height = extraction_dimensions(original, attacked, attack)
        gray = cv2.cvtColor(attacked, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (resize_width, resize_height), interpolation=cv2.INTER_LINEAR)
        feature_start = time.perf_counter()
        features = feature_matrix_dynamic(resized)
        feature_seconds = time.perf_counter() - feature_start
        prediction_start = time.perf_counter()
        prediction = model.predict(features).reshape(
            (resize_height - 31 + 1, resize_width - 31 + 1)
        ).astype(np.uint8)
        prediction_seconds = time.perf_counter() - prediction_start
        write_image(output_path, prediction)
        values = metrics(reference, prediction, attack)
        record = {
            "asset_id": sample_dir.name,
            "attack_family": attack["attack_family"],
            "attack_name": attack["attack_name"],
            "attack_parameters": attack["parameters"],
            "attack_path": attack["relative_output_path"],
            "extracted_width": int(prediction.shape[1]),
            "extracted_height": int(prediction.shape[0]),
            "feature_seconds": round(feature_seconds, 6),
            "prediction_seconds": round(prediction_seconds, 6),
            **values,
        }
        result_path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
        records.append(record)
    return records


def feature_matrix_dynamic(image: np.ndarray) -> np.ndarray:
    """Use the original fixed 31x31 feature window on a variable-size attacked image."""
    height, width = image.shape
    watermark_height, watermark_width = height - 31 + 1, width - 31 + 1
    mask = circle_mask_cached()
    features = np.empty((watermark_height * watermark_width, 7), dtype=np.float64)
    index = 0
    for row in range(watermark_height):
        for column in range(watermark_width):
            patch = image[row : row + 31, column : column + 31]
            circular_patch = np.zeros_like(patch)
            circular_patch[mask] = patch[mask]
            hu = cv2.HuMoments(cv2.moments(circular_patch)).flatten()
            features[index] = -np.sign(hu) * np.log10(np.abs(hu) + 1e-10)
            index += 1
    return features


_MASK: np.ndarray | None = None


def circle_mask_cached() -> np.ndarray:
    global _MASK
    if _MASK is None:
        center = 31 // 2
        y, x = np.ogrid[:31, :31]
        _MASK = (x - center) ** 2 + (y - center) ** 2 <= center**2
    return _MASK


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--processes", type=int, default=4)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    experiment = args.experiment.resolve()
    with (experiment / "manifest" / "attack_manifest.csv").open(encoding="utf-8", newline="") as file:
        manifest = list(csv.DictReader(file))
    grouped: dict[str, list[dict[str, str]]] = {}
    for record in manifest:
        grouped.setdefault(record["asset_id"], []).append(record)
    sample_dirs = [experiment / "samples" / asset_id for asset_id in sorted(grouped)]
    if args.limit is not None:
        sample_dirs = sample_dirs[: args.limit]
    tasks = [
        (str(experiment), str(sample_dir), grouped[sample_dir.name], args.force)
        for sample_dir in sample_dirs
    ]

    all_records: list[dict[str, object]] = []
    with mp.Pool(processes=args.processes) as pool:
        for index, records in enumerate(pool.imap_unordered(process_sample, tasks), start=1):
            all_records.extend(records)
            print(f"[{index}/{len(tasks)}] extracted {len(records)} attacks")

    results_dir = experiment / "results"
    results_dir.mkdir(exist_ok=True)
    fields = [
        "asset_id", "attack_family", "attack_name", "attack_parameters", "attack_path",
        "extracted_width", "extracted_height", "feature_seconds", "prediction_seconds",
        "ncc", "masked_ncc", "masked_ssim", "bit_error_rate", "iou", "coverage",
        "true_positive", "false_positive", "false_negative",
        "reference_offset_x", "reference_offset_y", "extracted_offset_x", "extracted_offset_y",
        "overlap_width", "overlap_height",
        "centered_ncc",
        "geometry_aligned_angle", "geometry_aligned_ncc", "geometry_aligned_centered_ncc",
        "geometry_aligned_ssim", "geometry_aligned_bit_error_rate", "geometry_aligned_iou",
        "geometry_aligned_coverage", "geometry_reference_offset_x", "geometry_reference_offset_y",
        "geometry_extracted_offset_x", "geometry_extracted_offset_y",
        "geometry_overlap_width", "geometry_overlap_height",
    ]
    with (results_dir / "attack_metrics.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(sorted(all_records, key=lambda item: (item["asset_id"], item["attack_family"], item["attack_name"])))


if __name__ == "__main__":
    main()
