"""Summarize controlled geometric responses under the fixed cNCC screen.

The screening rule is the rounded, protocol-calibrated threshold used in the
manuscript: centered_ncc >= 0.160.  This script deliberately leaves the
frozen perceptual-assessment summaries untouched; the two endpoints are
reported separately.
"""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path


TAU = 0.160
ROOT = Path(__file__).resolve().parents[1]


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Two-sided Wilson 95% interval for a binomial proportion."""
    if total <= 0:
        raise ValueError("total must be positive")
    proportion = successes / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(
        proportion * (1 - proportion) / total + z * z / (4 * total * total)
    ) / denominator
    return max(0.0, center - margin), min(1.0, center + margin)


def write_summary(input_path: Path, group_field: str, output_path: Path) -> None:
    grouped: dict[float, list[float]] = defaultdict(list)
    with input_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            grouped[float(row[group_field])].append(float(row["centered_ncc"]))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                group_field,
                "threshold",
                "n",
                "screening_passes",
                "screening_fails",
                "screening_pass_rate",
                "wilson_95_low",
                "wilson_95_high",
                "mean_centered_ncc",
            ],
        )
        writer.writeheader()
        for value in sorted(grouped):
            scores = grouped[value]
            successes = sum(score >= TAU for score in scores)
            low, high = wilson_interval(successes, len(scores))
            writer.writerow(
                {
                    group_field: value,
                    "threshold": TAU,
                    "n": len(scores),
                    "screening_passes": successes,
                    "screening_fails": len(scores) - successes,
                    "screening_pass_rate": f"{successes / len(scores):.9f}",
                    "wilson_95_low": f"{low:.9f}",
                    "wilson_95_high": f"{high:.9f}",
                    "mean_centered_ncc": f"{sum(scores) / len(scores):.9f}",
                }
            )


def main() -> None:
    write_summary(
        ROOT / "crop_spatial" / "results" / "crop_spatial_metrics.csv",
        "retain_percent",
        ROOT / "crop_spatial" / "results" / "cncc_screening_response_by_retain_percent.csv",
    )
    write_summary(
        ROOT / "scale" / "results" / "scale_metrics.csv",
        "parameter_value",
        ROOT / "scale" / "results" / "cncc_screening_response_by_factor.csv",
    )


if __name__ == "__main__":
    main()
