"""Summarize all controlled attacks whose extracted watermarks remain 70 x 70.

The output intentionally excludes crop and rotation: their support or orientation
changes require a separate geometric treatment and are not direct SSIM/BER cases.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path


SUITE = Path(__file__).resolve().parents[1]
CONDITIONS = (
    "scale", "gaussian_noise", "speckle_noise", "salt_pepper_noise",
    "jpeg", "jpeg2000", "mean_filter", "median_filter", "gaussian_filter",
)
FIELDS = [
    "condition", "parameter_name", "parameter_value", "n", "mean_ssim", "sd_ssim",
    "mean_ber", "sd_ber", "mean_centered_ncc", "mean_iou",
]


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def sample_sd(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    center = mean(values)
    return math.sqrt(sum((value - center) ** 2 for value in values) / (len(values) - 1))


def main() -> None:
    rows: list[dict[str, object]] = []
    for condition in CONDITIONS:
        path = SUITE / condition / "results" / f"{condition}_metrics.csv"
        with path.open(encoding="utf-8", newline="") as handle:
            data = list(csv.DictReader(handle))
        if not data:
            raise RuntimeError(f"No metrics found: {path}")
        if any(row["extracted_width"] != "70" or row["extracted_height"] != "70" for row in data):
            raise RuntimeError(f"Non-direct extracted size found in {condition}")
        groups: dict[tuple[str, str], list[dict[str, str]]] = {}
        for row in data:
            groups.setdefault((row["parameter_name"], row["parameter_value"]), []).append(row)
        for (parameter_name, parameter_value), group in groups.items():
            ssim = [float(row["masked_ssim"]) for row in group]
            ber = [float(row["bit_error_rate"]) for row in group]
            ncc = [float(row["centered_ncc"]) for row in group]
            iou = [float(row["iou"]) for row in group]
            rows.append({
                "condition": condition,
                "parameter_name": parameter_name,
                "parameter_value": float(parameter_value),
                "n": len(group),
                "mean_ssim": mean(ssim), "sd_ssim": sample_sd(ssim),
                "mean_ber": mean(ber), "sd_ber": sample_sd(ber),
                "mean_centered_ncc": mean(ncc), "mean_iou": mean(iou),
            })
    rows.sort(key=lambda row: (str(row["condition"]), float(row["parameter_value"])))
    output = SUITE / "direct_metric_summary"
    output.mkdir(exist_ok=True)
    with (output / "direct_metric_response_by_level.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader(); writer.writerows(rows)
    with (output / "README.md").open("w", encoding="utf-8") as handle:
        handle.write(
            "# Direct metric controlled-response summary\n\n"
            "This table summarizes single-operation controlled attacks for which every "
            "extracted watermark is 70 x 70 and can be compared directly with its "
            "registered 70 x 70 reference. It includes scale (after the prescribed "
            "physical-resolution restoration), Gaussian/speckle/salt-and-pepper noise, "
            "JPEG/JPEG2000 compression, and mean/median/Gaussian filtering. Crop and "
            "rotation are intentionally excluded because they require support or "
            "orientation handling rather than direct full-watermark metrics.\n"
        )
    print(f"wrote {len(rows)} level summaries to {output}")


if __name__ == "__main__":
    main()
