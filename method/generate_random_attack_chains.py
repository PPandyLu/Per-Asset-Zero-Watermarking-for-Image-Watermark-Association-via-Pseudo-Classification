"""Generate a reproducible 5,000-sample stochastic compound-attack study."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from generate_attacks import (
    crop,
    gaussian_noise,
    jpeg2000_parameter,
    read_image,
    rotate,
    salt_pepper_noise,
    scale,
    speckle_noise,
    write_image,
)


SEED = 20260829
CHAINS_PER_ASSET = 25
# Every asset receives exactly this multiset in shuffled order.
CHAIN_LENGTHS = [1] * 15 + [2] * 6 + [3] * 3 + [4]

# The ten distinct real-world processing families.  A chain may use a family once.
FAMILY_ORDER = {
    "rotation": 0,
    "scale": 0,
    "crop": 0,
    "mean_filter": 1,
    "median_filter": 1,
    "gaussian_filter": 1,
    "gaussian_noise": 2,
    "speckle_noise": 2,
    "salt_pepper_noise": 2,
    "compression": 3,
}

STRENGTHS = ("light", "moderate", "severe")
STRENGTH_WEIGHTS = (0.55, 0.30, 0.15)

PARAMETERS = {
    "rotation": {
        "light": ({"angle": 15},),
        "moderate": ({"angle": 45}, {"angle": 90}),
        "severe": ({"angle": 135}, {"angle": 180}, {"angle": 270}),
    },
    "scale": {
        "light": ({"factor": 0.75}, {"factor": 1.25}),
        "moderate": ({"factor": 0.50}, {"factor": 1.50}),
        "severe": ({"factor": 0.30}, {"factor": 2.00}),
    },
    "crop": {
        "light": ({"retain_percent": 90},),
        "moderate": ({"retain_percent": 70},),
        "severe": ({"retain_percent": 50}, {"retain_percent": 30}),
    },
    "mean_filter": {
        "light": ({"kernel": 3},),
        "moderate": ({"kernel": 7},),
        "severe": ({"kernel": 11}, {"kernel": 15}, {"kernel": 19}),
    },
    "median_filter": {
        "light": ({"kernel": 3},),
        "moderate": ({"kernel": 7},),
        "severe": ({"kernel": 11}, {"kernel": 15}, {"kernel": 19}),
    },
    "gaussian_filter": {
        "light": ({"kernel": 3},),
        "moderate": ({"kernel": 7},),
        "severe": ({"kernel": 11}, {"kernel": 15}, {"kernel": 19}),
    },
    "gaussian_noise": {
        "light": ({"sigma": 5},),
        "moderate": ({"sigma": 15},),
        "severe": ({"sigma": 30}, {"sigma": 50}),
    },
    "speckle_noise": {
        "light": ({"sigma": 0.05},),
        "moderate": ({"sigma": 0.10},),
        "severe": ({"sigma": 0.20}, {"sigma": 0.30}, {"sigma": 0.50}),
    },
    "salt_pepper_noise": {
        "light": ({"density": 0.01},),
        "moderate": ({"density": 0.03}, {"density": 0.05}),
        "severe": ({"density": 0.08}, {"density": 0.10}),
    },
    "compression": {
        "light": ({"codec": "jpeg", "quality_factor": 90}, {"codec": "jpeg2000", "compression_ratio": 5}),
        "moderate": ({"codec": "jpeg", "quality_factor": 70}, {"codec": "jpeg2000", "compression_ratio": 10}),
        "severe": (
            {"codec": "jpeg", "quality_factor": 50},
            {"codec": "jpeg", "quality_factor": 30},
            {"codec": "jpeg", "quality_factor": 10},
            {"codec": "jpeg2000", "compression_ratio": 20},
            {"codec": "jpeg2000", "compression_ratio": 40},
            {"codec": "jpeg2000", "compression_ratio": 80},
        ),
    },
}


def chain_seed(asset_id: str, chain_index: int) -> int:
    text = f"{SEED}:{asset_id}:{chain_index}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(text).digest()[:8], "little")


def choose_operation(family: str, rng: random.Random) -> dict[str, object]:
    strength = rng.choices(STRENGTHS, weights=STRENGTH_WEIGHTS, k=1)[0]
    parameters = dict(rng.choice(PARAMETERS[family][strength]))
    return {"family": family, "strength": strength, "parameters": parameters}


def build_chain(asset_id: str, chain_index: int, length: int) -> tuple[int, list[dict[str, object]]]:
    seed = chain_seed(asset_id, chain_index)
    rng = random.Random(seed)
    families = rng.sample(list(FAMILY_ORDER), k=length)
    operations = [choose_operation(family, rng) for family in families]
    # Encoding is terminal; the remaining stages follow a physically plausible order.
    operations.sort(key=lambda operation: FAMILY_ORDER[str(operation["family"])])
    return seed, operations


def apply_operation(
    image: np.ndarray, operation: dict[str, object], rng: np.random.Generator
) -> tuple[np.ndarray, dict[str, object], list[int] | None, str]:
    family = str(operation["family"])
    parameters = dict(operation["parameters"])
    details: dict[str, object] = {}
    if family == "rotation":
        return rotate(image, float(parameters["angle"])), details, None, ".png"
    if family == "scale":
        return scale(image, float(parameters["factor"])), details, None, ".png"
    if family == "crop":
        output, details = crop(image, int(parameters["retain_percent"]), rng)
        return output, details, None, ".png"
    if family == "mean_filter":
        kernel = int(parameters["kernel"])
        return cv2.boxFilter(image, -1, (kernel, kernel), normalize=True), details, None, ".png"
    if family == "median_filter":
        return cv2.medianBlur(image, int(parameters["kernel"])), details, None, ".png"
    if family == "gaussian_filter":
        kernel = int(parameters["kernel"])
        return cv2.GaussianBlur(image, (kernel, kernel), 0), details, None, ".png"
    if family == "gaussian_noise":
        return gaussian_noise(image, float(parameters["sigma"]), rng), details, None, ".png"
    if family == "speckle_noise":
        return speckle_noise(image, float(parameters["sigma"]), rng), details, None, ".png"
    if family == "salt_pepper_noise":
        return salt_pepper_noise(image, float(parameters["density"]), rng), details, None, ".png"
    if family == "compression":
        if parameters["codec"] == "jpeg":
            return image, details, [cv2.IMWRITE_JPEG_QUALITY, int(parameters["quality_factor"])], ".jpg"
        return image, details, [
            cv2.IMWRITE_JPEG2000_COMPRESSION_X1000,
            jpeg2000_parameter(int(parameters["compression_ratio"])),
        ], ".jp2"
    raise ValueError(f"Unknown attack family: {family}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=Path, required=True)
    args = parser.parse_args()
    experiment = args.experiment.resolve()
    sample_dirs = sorted(path for path in (experiment / "samples").iterdir() if path.is_dir())
    if not sample_dirs:
        raise RuntimeError("No sample directories found.")

    records: list[dict[str, object]] = []
    family_counts: Counter[str] = Counter()
    for sample_number, sample_dir in enumerate(sample_dirs, start=1):
        asset_id = sample_dir.name
        image = read_image(sample_dir / "original.png")
        lengths = list(CHAIN_LENGTHS)
        random.Random(chain_seed(asset_id, 0)).shuffle(lengths)
        for chain_index, length in enumerate(lengths, start=1):
            seed, operations = build_chain(asset_id, chain_index, length)
            attacked = image
            applied_operations = []
            final_encoding: list[int] | None = None
            suffix = ".png"
            for operation_index, operation in enumerate(operations, start=1):
                operation_rng = np.random.default_rng(seed + operation_index)
                attacked, details, encoding, operation_suffix = apply_operation(attacked, operation, operation_rng)
                applied_operations.append({**operation, "derived_parameters": details})
                family_counts[str(operation["family"])] += 1
                if encoding is not None:
                    final_encoding, suffix = encoding, operation_suffix
            chain_name = f"chain_{chain_index:03d}"
            output_path = sample_dir / "attacks" / "random_chains" / f"{chain_name}{suffix}"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            if not output_path.exists():
                write_image(output_path, attacked, final_encoding)
            records.append(
                {
                    "asset_id": asset_id,
                    "attack_family": "random_chain",
                    "attack_name": chain_name,
                    "parameters": json.dumps(
                        {"seed": seed, "chain_length": length, "operations": applied_operations},
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                    "derived_parameters": json.dumps(applied_operations, ensure_ascii=False, sort_keys=True),
                    "relative_output_path": str(output_path.relative_to(experiment)),
                    "width": attacked.shape[1],
                    "height": attacked.shape[0],
                    "bytes": output_path.stat().st_size,
                }
            )
        print(f"[{sample_number}/{len(sample_dirs)}] {asset_id}: {CHAINS_PER_ASSET} random chains")

    manifest_dir = experiment / "manifest"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    with (manifest_dir / "attack_manifest.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    config = {
        "seed": SEED,
        "assets": len(sample_dirs),
        "chains_per_asset": CHAINS_PER_ASSET,
        "chain_length_counts": dict(Counter(CHAIN_LENGTHS)),
        "total_chains": len(records),
        "total_operations": sum(family_counts.values()),
        "family_operation_counts": dict(sorted(family_counts.items())),
        "strength_weights": dict(zip(STRENGTHS, STRENGTH_WEIGHTS)),
        "family_order": FAMILY_ORDER,
        "parameter_sets": PARAMETERS,
    }
    (manifest_dir / "attack_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
