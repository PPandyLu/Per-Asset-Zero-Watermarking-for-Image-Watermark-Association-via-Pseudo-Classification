"""Generate deterministic single and compound attacks for the xBD batch experiment."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


ATTACKS = [
    ("rotation", "deg_015", {"angle": 15}),
    ("rotation", "deg_045", {"angle": 45}),
    ("rotation", "deg_090", {"angle": 90}),
    ("rotation", "deg_135", {"angle": 135}),
    ("rotation", "deg_180", {"angle": 180}),
    ("rotation", "deg_270", {"angle": 270}),
    ("scale", "factor_0.30", {"factor": 0.30}),
    ("scale", "factor_0.50", {"factor": 0.50}),
    ("scale", "factor_0.75", {"factor": 0.75}),
    ("scale", "factor_1.25", {"factor": 1.25}),
    ("scale", "factor_1.50", {"factor": 1.50}),
    ("scale", "factor_2.00", {"factor": 2.00}),
    *[
        ("crop", f"retain_{retain:02d}_rep_{rep}", {"retain_percent": retain, "replicate": rep})
        for retain in (90, 70, 50, 30)
        for rep in range(1, 4)
    ],
    *[
        (kind, f"kernel_{kernel:02d}", {"kernel": kernel})
        for kind in ("mean_filter", "median_filter", "gaussian_filter")
        for kernel in (3, 7, 11, 15, 19)
    ],
    *[("gaussian_noise", f"sigma_{sigma:02d}", {"sigma": sigma}) for sigma in (5, 15, 30, 50)],
    *[("speckle_noise", f"sigma_{sigma:.2f}", {"sigma": sigma}) for sigma in (0.05, 0.10, 0.20, 0.30, 0.50)],
    *[("salt_pepper_noise", f"density_{density:.2f}", {"density": density}) for density in (0.01, 0.03, 0.05, 0.08, 0.10)],
    *[("jpeg", f"qf_{quality:02d}", {"quality_factor": quality}) for quality in (90, 70, 50, 30, 10)],
    *[("jpeg2000", f"cr_{ratio:02d}", {"compression_ratio": ratio}) for ratio in (5, 10, 20, 40, 80)],
    ("compound", "rotate_045__gaussian_sigma_15", {"operations": ["rotate_045", "gaussian_sigma_15"]}),
    ("compound", "crop_retain_70__jpeg_qf_30", {"operations": ["crop_retain_70", "jpeg_qf_30"]}),
    ("compound", "rotate_045__scale_0.50__jpeg_qf_30", {"operations": ["rotate_045", "scale_0.50", "jpeg_qf_30"]}),
]


def read_image(path: Path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Could not decode {path}")
    return image


def write_image(path: Path, image: np.ndarray, parameters: list[int] | None = None) -> None:
    success, encoded = cv2.imencode(path.suffix, image, parameters or [])
    if not success:
        raise RuntimeError(f"Could not encode {path}")
    encoded.tofile(path)


def seed(asset_id: str, attack_name: str) -> int:
    return int.from_bytes(
        hashlib.sha256(f"{asset_id}:{attack_name}".encode("utf-8")).digest()[:8], "little"
    )


def rotate(image: np.ndarray, angle: float) -> np.ndarray:
    height, width = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
    new_width = round(height * abs(matrix[0, 1]) + width * abs(matrix[0, 0]))
    new_height = round(height * abs(matrix[0, 0]) + width * abs(matrix[0, 1]))
    matrix[0, 2] += (new_width - width) / 2
    matrix[1, 2] += (new_height - height) / 2
    return cv2.warpAffine(image, matrix, (new_width, new_height), flags=cv2.INTER_NEAREST)


def scale(image: np.ndarray, factor: float) -> np.ndarray:
    height, width = image.shape[:2]
    return cv2.resize(
        image,
        (max(1, round(width * factor)), max(1, round(height * factor))),
        interpolation=cv2.INTER_NEAREST,
    )


def crop(image: np.ndarray, retain_percent: int, rng: np.random.Generator) -> tuple[np.ndarray, dict[str, int]]:
    height, width = image.shape[:2]
    side_ratio = (retain_percent / 100) ** 0.5
    crop_width, crop_height = round(width * side_ratio), round(height * side_ratio)
    x = int(rng.integers(0, width - crop_width + 1))
    y = int(rng.integers(0, height - crop_height + 1))
    return image[y : y + crop_height, x : x + crop_width], {
        "x": x, "y": y, "width": crop_width, "height": crop_height
    }


def gaussian_noise(image: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    noise = rng.normal(0, sigma, image.shape)
    return np.clip(image.astype(np.float64) + noise, 0, 255).astype(np.uint8)


def speckle_noise(image: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    noise = rng.normal(0, sigma, image.shape)
    return np.clip(image.astype(np.float64) + image.astype(np.float64) * noise, 0, 255).astype(np.uint8)


def salt_pepper_noise(image: np.ndarray, density: float, rng: np.random.Generator) -> np.ndarray:
    output = image.copy()
    mask = rng.random(image.shape[:2])
    output[mask < density / 2] = 0
    output[mask > 1 - density / 2] = 255
    return output


def jpeg2000_parameter(compression_ratio: int) -> int:
    """OpenCV's X1000 control approximates the retained bitrate fraction."""
    return max(1, min(1000, round(1000 / compression_ratio)))


def apply_attack(
    image: np.ndarray, asset_id: str, kind: str, name: str, parameters: dict[str, object]
) -> tuple[np.ndarray, dict[str, object], list[int] | None]:
    rng = np.random.default_rng(seed(asset_id, f"{kind}/{name}"))
    details: dict[str, object] = {}
    encoding: list[int] | None = None
    if kind == "rotation":
        return rotate(image, float(parameters["angle"])), details, encoding
    if kind == "scale":
        return scale(image, float(parameters["factor"])), details, encoding
    if kind == "crop":
        output, details = crop(image, int(parameters["retain_percent"]), rng)
        return output, details, encoding
    if kind == "mean_filter":
        kernel = int(parameters["kernel"])
        return cv2.boxFilter(image, -1, (kernel, kernel), normalize=True), details, encoding
    if kind == "median_filter":
        return cv2.medianBlur(image, int(parameters["kernel"])), details, encoding
    if kind == "gaussian_filter":
        kernel = int(parameters["kernel"])
        return cv2.GaussianBlur(image, (kernel, kernel), 0), details, encoding
    if kind == "gaussian_noise":
        return gaussian_noise(image, float(parameters["sigma"]), rng), details, encoding
    if kind == "speckle_noise":
        return speckle_noise(image, float(parameters["sigma"]), rng), details, encoding
    if kind == "salt_pepper_noise":
        return salt_pepper_noise(image, float(parameters["density"]), rng), details, encoding
    if kind == "jpeg":
        encoding = [cv2.IMWRITE_JPEG_QUALITY, int(parameters["quality_factor"])]
        return image, details, encoding
    if kind == "jpeg2000":
        encoding = [cv2.IMWRITE_JPEG2000_COMPRESSION_X1000, jpeg2000_parameter(int(parameters["compression_ratio"]))]
        return image, details, encoding
    if kind == "compound":
        if name == "rotate_045__gaussian_sigma_15":
            return gaussian_noise(rotate(image, 45), 15, rng), details, encoding
        if name == "crop_retain_70__jpeg_qf_30":
            output, details = crop(image, 70, rng)
            encoding = [cv2.IMWRITE_JPEG_QUALITY, 30]
            return output, details, encoding
        if name == "rotate_045__scale_0.50__jpeg_qf_30":
            output = scale(rotate(image, 45), 0.50)
            encoding = [cv2.IMWRITE_JPEG_QUALITY, 30]
            return output, details, encoding
    raise ValueError(f"Unknown attack: {kind}/{name}")


def extension(kind: str, name: str) -> str:
    if kind == "jpeg" or (kind == "compound" and name != "rotate_045__gaussian_sigma_15"):
        return ".jpg"
    if kind == "jpeg2000":
        return ".jp2"
    return ".png"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    experiment = args.experiment.resolve()
    sample_dirs = sorted(path for path in (experiment / "samples").iterdir() if path.is_dir())
    if args.limit is not None:
        sample_dirs = sample_dirs[: args.limit]
    records: list[dict[str, object]] = []

    for sample_index, sample_dir in enumerate(sample_dirs, start=1):
        asset_id = sample_dir.name
        image = read_image(sample_dir / "original.png")
        for kind, name, parameters in ATTACKS:
            output_path = sample_dir / "attacks" / kind / f"{name}{extension(kind, name)}"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            if not output_path.exists():
                attacked, details, encoding = apply_attack(image, asset_id, kind, name, parameters)
                write_image(output_path, attacked, encoding)
            else:
                attacked = read_image(output_path)
                details = {}
            records.append(
                {
                    "asset_id": asset_id,
                    "attack_family": kind,
                    "attack_name": name,
                    "parameters": json.dumps(parameters, sort_keys=True),
                    "derived_parameters": json.dumps(details, sort_keys=True),
                    "relative_output_path": str(output_path.relative_to(experiment)),
                    "width": attacked.shape[1],
                    "height": attacked.shape[0],
                    "bytes": output_path.stat().st_size,
                }
            )
        print(f"[{sample_index}/{len(sample_dirs)}] {asset_id}: {len(ATTACKS)} attacks")

    manifest_dir = experiment / "manifest"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    with (manifest_dir / "attack_manifest.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    (manifest_dir / "attack_config.json").write_text(
        json.dumps({"attacks_per_sample": len(ATTACKS), "attacks": ATTACKS}, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
