"""Group-disjoint calibration/evaluation of objective reconstruction metrics.

This analysis evaluates a protocol-specific centered-NCC screening rule using
already archived task-level metrics. Calibration and evaluation are separated
by registered source image.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
POSITIVE_PATH = ROOT / "inputs" / "positive.csv"
NEGATIVE_PATH = ROOT / "inputs" / "negative.csv"
OUT = ROOT / "recomputed"


METRICS = {
    "centered_ncc": 1.0,
    "masked_ssim": 1.0,
    "bit_error_rate": -1.0,
    "iou": 1.0,
    "coverage": 1.0,
}


def hashed_order(seed: str, event: str, asset: str) -> str:
    return hashlib.sha256(f"{seed}|{event}|{asset}".encode()).hexdigest()


def calibration_assets(frame: pd.DataFrame, seed: str, fraction: float = 0.2) -> set[str]:
    chosen: set[str] = set()
    for event, subset in frame[["event", "group_asset"]].drop_duplicates().groupby("event"):
        ordered = sorted(
            subset["group_asset"].tolist(),
            key=lambda asset: hashed_order(seed, str(event), str(asset)),
        )
        count = max(1, math.ceil(len(ordered) * fraction))
        chosen.update(ordered[:count])
    return chosen


def auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Mann--Whitney AUC with average ranks for ties."""
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=float)
    index = 0
    while index < len(scores):
        end = index + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[index]:
            end += 1
        ranks[order[index:end]] = (index + 1 + end) / 2.0
        index = end
    positives = labels == 1
    n_pos = int(positives.sum())
    n_neg = int((~positives).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[positives].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def select_threshold(labels: np.ndarray, scores: np.ndarray) -> float:
    """Threshold maximizing Youden's J; positive predictions use score >= threshold."""
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    order = np.argsort(-scores, kind="mergesort")
    labels = labels[order]
    scores = scores[order]
    n_pos = int(labels.sum())
    n_neg = len(labels) - n_pos
    true_pos = false_pos = 0
    best_j = -float("inf")
    best_threshold = float(scores[0])
    index = 0
    while index < len(scores):
        end = index + 1
        while end < len(scores) and scores[end] == scores[index]:
            end += 1
        group = labels[index:end]
        true_pos += int(group.sum())
        false_pos += len(group) - int(group.sum())
        tpr = true_pos / n_pos if n_pos else 0.0
        fpr = false_pos / n_neg if n_neg else 0.0
        score_j = tpr - fpr
        if score_j > best_j:
            best_j = score_j
            best_threshold = float(scores[index])
        index = end
    return best_threshold


def operating_point(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float]:
    labels = np.asarray(labels, dtype=int)
    predicted = np.asarray(scores, dtype=float) >= threshold
    tp = int(np.sum((labels == 1) & predicted))
    fn = int(np.sum((labels == 1) & ~predicted))
    fp = int(np.sum((labels == 0) & predicted))
    tn = int(np.sum((labels == 0) & ~predicted))
    return {
        "threshold": threshold,
        "tpr": tp / (tp + fn) if tp + fn else float("nan"),
        "frr": fn / (tp + fn) if tp + fn else float("nan"),
        "far": fp / (fp + tn) if fp + tn else float("nan"),
        "tnr": tn / (fp + tn) if fp + tn else float("nan"),
        "balanced_accuracy": ((tp / (tp + fn)) + (tn / (fp + tn))) / 2 if (tp + fn and fp + tn) else float("nan"),
    }


def fit_logistic(calibration: pd.DataFrame, metric_names: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = calibration[metric_names].to_numpy(dtype=float)
    medians = np.nanmedian(values, axis=0)
    values = np.where(np.isnan(values), medians, values)
    means = values.mean(axis=0)
    scales = values.std(axis=0)
    scales[scales == 0] = 1.0
    design = np.column_stack([np.ones(len(values)), (values - means) / scales])
    labels = calibration["label"].to_numpy(dtype=float)
    pos_weight = len(labels) / (2 * max(1, labels.sum()))
    neg_weight = len(labels) / (2 * max(1, (1 - labels).sum()))
    weights = np.where(labels == 1, pos_weight, neg_weight)
    beta = np.zeros(design.shape[1], dtype=float)
    for _ in range(5000):
        logits = np.clip(design @ beta, -35, 35)
        probabilities = 1.0 / (1.0 + np.exp(-logits))
        gradient = design.T @ ((probabilities - labels) * weights) / len(labels)
        updated = beta - 0.08 * gradient
        if np.max(np.abs(updated - beta)) < 1e-9:
            beta = updated
            break
        beta = updated
    return beta, medians, np.column_stack([means, scales])


def logistic_scores(frame: pd.DataFrame, metric_names: list[str], beta: np.ndarray, medians: np.ndarray, scaler: np.ndarray) -> np.ndarray:
    values = frame[metric_names].to_numpy(dtype=float)
    values = np.where(np.isnan(values), medians, values)
    design = np.column_stack([np.ones(len(values)), (values - scaler[:, 0]) / scaler[:, 1]])
    logits = np.clip(design @ beta, -35, 35)
    return 1.0 / (1.0 + np.exp(-logits))


def parse_operations(serialized: str) -> set[str]:
    try:
        payload = json.loads(serialized)
    except (json.JSONDecodeError, TypeError):
        return set()
    return {str(operation.get("family", "")) for operation in payload.get("operations", [])}


def make_frame() -> pd.DataFrame:
    positive = pd.read_csv(POSITIVE_PATH)
    positive = positive.rename(columns={"asset_id": "group_asset"})
    positive["event"] = positive["group_asset"].str.extract(r"xbd-(?:hold|test)-(.+)_\d+_pre_disaster")[0].fillna("unknown")
    positive["label"] = 1
    positive["source"] = "positive"
    positive["input_condition"] = "attacked"
    negative = pd.read_csv(NEGATIVE_PATH)
    negative = negative.rename(columns={"target_asset_id": "group_asset", "target_event": "event"})
    negative["label"] = 0
    negative["source"] = "negative"
    frames = []
    for frame in (positive, negative):
        if "relationship" not in frame:
            frame["relationship"] = "positive_control"
        keep = ["group_asset", "event", "label", "source", "relationship", "input_condition", "attack_parameters", *METRICS]
        local = frame[keep].copy()
        local["operations"] = local["attack_parameters"].map(parse_operations)
        local["has_crop"] = local["operations"].map(lambda values: "crop" in values)
        local["has_rotation"] = local["operations"].map(lambda values: "rotation" in values)
        local["full_support"] = local["coverage"].fillna(0) >= 0.999999
        frames.append(local)
    return pd.concat(frames, ignore_index=True)


def evaluate_split(frame: pd.DataFrame, seed: str) -> list[dict[str, object]]:
    calibration_ids = calibration_assets(frame, seed)
    calibration = frame[frame["group_asset"].isin(calibration_ids)].copy()
    test = frame[~frame["group_asset"].isin(calibration_ids)].copy()
    rows: list[dict[str, object]] = []
    for metric, direction in METRICS.items():
        cal = calibration[["label", metric]].dropna()
        held = test[["label", metric]].dropna()
        cal_scores = cal[metric].to_numpy(float) * direction
        held_scores = held[metric].to_numpy(float) * direction
        threshold = select_threshold(cal["label"].to_numpy(int), cal_scores)
        point = operating_point(held["label"].to_numpy(int), held_scores, threshold)
        rows.append({
            "split": seed,
            "rule": metric,
            "kind": "single_metric",
            "calibration_tasks": len(cal),
            "test_tasks": len(held),
            "calibration_auc": auc(cal["label"].to_numpy(int), cal_scores),
            "test_auc": auc(held["label"].to_numpy(int), held_scores),
            **point,
        })
    metric_names = list(METRICS)
    beta, medians, scaler = fit_logistic(calibration, metric_names)
    cal_scores = logistic_scores(calibration, metric_names, beta, medians, scaler)
    held_scores = logistic_scores(test, metric_names, beta, medians, scaler)
    threshold = select_threshold(calibration["label"].to_numpy(int), cal_scores)
    point = operating_point(test["label"].to_numpy(int), held_scores, threshold)
    rows.append({
        "split": seed,
        "rule": "objective_logistic_combination",
        "kind": "exploratory_combination",
        "calibration_tasks": len(calibration),
        "test_tasks": len(test),
        "calibration_auc": auc(calibration["label"].to_numpy(int), cal_scores),
        "test_auc": auc(test["label"].to_numpy(int), held_scores),
        **point,
    })
    return rows


def stratum_auc(frame: pd.DataFrame, split_seed: str) -> pd.DataFrame:
    calibration_ids = calibration_assets(frame, split_seed)
    test = frame[~frame["group_asset"].isin(calibration_ids)].copy()
    strata = {
        "all_test_tasks": pd.Series(True, index=test.index),
        "full_support": test["full_support"],
        "partial_support": ~test["full_support"],
        "positive_crop_tasks_plus_all_negatives": (test["label"] == 0) | ((test["label"] == 1) & test["has_crop"]),
        "positive_rotation_tasks_plus_all_negatives": (test["label"] == 0) | ((test["label"] == 1) & test["has_rotation"]),
    }
    rows: list[dict[str, object]] = []
    for name, mask in strata.items():
        subset = test[mask].copy()
        for metric, direction in METRICS.items():
            valid = subset[["label", metric]].dropna()
            rows.append({
                "stratum": name,
                "metric": metric,
                "tasks": len(valid),
                "positive_tasks": int(valid["label"].sum()),
                "negative_tasks": int((valid["label"] == 0).sum()),
                "auc": auc(valid["label"].to_numpy(int), valid[metric].to_numpy(float) * direction),
            })
    return pd.DataFrame(rows)


def centered_ncc_condition_performance(frame: pd.DataFrame, split_seed: str, threshold: float) -> pd.DataFrame:
    calibration_ids = calibration_assets(frame, split_seed)
    test = frame[~frame["group_asset"].isin(calibration_ids)].copy()
    test["predicted_match"] = test["centered_ncc"] >= threshold
    rows: list[dict[str, object]] = []
    positive = test[test["label"] == 1]
    positive_conditions = {
        "all_positive": pd.Series(True, index=positive.index),
        "positive_full_support": positive["full_support"],
        "positive_partial_support": ~positive["full_support"],
        "positive_contains_crop": positive["has_crop"],
        "positive_contains_rotation": positive["has_rotation"],
        "positive_crop_and_rotation": positive["has_crop"] & positive["has_rotation"],
    }
    for name, mask in positive_conditions.items():
        subset = positive[mask]
        rows.append({
            "class": "positive",
            "condition": name,
            "tasks": len(subset),
            "accepted": int(subset["predicted_match"].sum()),
            "rate": float(subset["predicted_match"].mean()) if len(subset) else float("nan"),
            "threshold": threshold,
        })
    negative = test[test["label"] == 0]
    for name, subset in [("all_negative", negative), *[(f"negative_{value}", group) for value, group in negative.groupby("input_condition")], *[(f"negative_{value}", group) for value, group in negative.groupby("relationship")]]:
        rows.append({
            "class": "negative",
            "condition": name,
            "tasks": len(subset),
            "accepted": int(subset["predicted_match"].sum()),
            "rate": float(subset["predicted_match"].mean()) if len(subset) else float("nan"),
            "threshold": threshold,
        })
    return pd.DataFrame(rows)


def fixed_threshold_stability(frame: pd.DataFrame, threshold: float) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for index in range(10):
        seed = f"objective_rule_repeat_{index}"
        calibration_ids = calibration_assets(frame, seed)
        test = frame[~frame["group_asset"].isin(calibration_ids)]
        point = operating_point(test["label"].to_numpy(int), test["centered_ncc"].to_numpy(float), threshold)
        rows.append({"split": seed, "threshold": threshold, **point})
    return pd.DataFrame(rows)


def markdown_summary(frame: pd.DataFrame, repeat_summary: pd.DataFrame, strata: pd.DataFrame, conditions: pd.DataFrame, fixed_summary: pd.Series) -> str:
    counts = frame.groupby("label").agg(tasks=("label", "size"), source_images=("group_asset", "nunique"))
    lines = [
        "# Exploratory objective-rule analysis",
        "",
        "This analysis evaluates a protocol-specific objective screening rule from archived reconstruction metrics; it does not establish a universal ownership threshold.",
        "",
        "## Data and split design",
        "",
        f"- Positive tasks: {int(counts.loc[1, 'tasks'])} from {int(counts.loc[1, 'source_images'])} registered source images.",
        f"- Negative tasks: {int(counts.loc[0, 'tasks'])} from {int(counts.loc[0, 'source_images'])} registered source images.",
        "- Ten deterministic 20% calibration splits were selected within event strata at the registered-source-image level. No source image appears in both calibration and held-out evaluation within a split.",
        "- Each continuous metric used a calibration-set Youden-J threshold. BER was inverted so that a larger analysis score always favors a match.",
        "- The multimetric logistic result is exploratory only; it is included to test whether a simple objective combination appears more stable than individual metrics.",
        "",
        "## Held-out performance across ten source-image-disjoint splits",
        "",
        "| Rule | Held-out AUC, median [min, max] | Balanced accuracy, median [min, max] | TPR median | FAR median |",
        "|---|---:|---:|---:|---:|",
    ]
    ordered = repeat_summary.sort_values("test_auc_median", ascending=False)
    for _, row in ordered.iterrows():
        lines.append(
            f"| {row['rule']} | {row['test_auc_median']:.3f} [{row['test_auc_min']:.3f}, {row['test_auc_max']:.3f}] | "
            f"{row['balanced_accuracy_median']:.3f} [{row['balanced_accuracy_min']:.3f}, {row['balanced_accuracy_max']:.3f}] | "
            f"{row['tpr_median']:.3f} | {row['far_median']:.3f} |"
        )
    centered = repeat_summary.loc[repeat_summary["rule"] == "centered_ncc"].iloc[0]
    lines.extend([
        "",
        f"The centered-NCC calibration threshold varied from {centered['threshold_min']:.3f} to {centered['threshold_max']:.3f} across the ten splits (median {centered['threshold_median']:.3f}).",
        f"Using the rounded candidate threshold of 0.160 on each held-out split gave median TPR {fixed_summary['tpr_median']:.3f}, FAR {fixed_summary['far_median']:.3f}, and balanced accuracy {fixed_summary['balanced_accuracy_median']:.3f}.",
        "",
        "## Primary-split centered-NCC operating point by condition",
        "",
        "| Class and condition | Tasks | Accepted | Rate |",
        "|---|---:|---:|---:|",
    ])
    for _, row in conditions.iterrows():
        lines.append(f"| {row['condition']} | {int(row['tasks'])} | {int(row['accepted'])} | {row['rate']:.3f} |")
    lines.extend([
        "",
        "## Interpretation guardrails",
        "",
        "- A stable score in this table indicates discrimination in the archived experimental tasks; it does not by itself validate a universal ownership rule.",
        "- Coverage and masked metrics depend on the overlap procedure. Rotation and incomplete-support cases remain separate strata because their values combine extraction quality with geometric support and alignment.",
        "- Any decision to promote a metric rule would require a preregistered calibration procedure, a final untouched test cohort, and an operational definition of valid support.",
        "",
        "The machine-readable outputs are `repeat_split_results.csv`, `repeat_split_summary.csv`, `stratified_auc_primary_split.csv`, and `centered_ncc_condition_performance_primary_split.csv`.",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    frame = make_frame()
    frame.to_csv(OUT / "analysis_task_index.csv", index=False)
    repeats = []
    for index in range(10):
        repeats.extend(evaluate_split(frame, f"objective_rule_repeat_{index}"))
    repeat_results = pd.DataFrame(repeats)
    repeat_results.to_csv(OUT / "repeat_split_results.csv", index=False)
    summary = repeat_results.groupby("rule").agg(
        test_auc_median=("test_auc", "median"),
        test_auc_min=("test_auc", "min"),
        test_auc_max=("test_auc", "max"),
        balanced_accuracy_median=("balanced_accuracy", "median"),
        balanced_accuracy_min=("balanced_accuracy", "min"),
        balanced_accuracy_max=("balanced_accuracy", "max"),
        tpr_median=("tpr", "median"),
        far_median=("far", "median"),
        threshold_median=("threshold", "median"),
        threshold_min=("threshold", "min"),
        threshold_max=("threshold", "max"),
    ).reset_index()
    summary.to_csv(OUT / "repeat_split_summary.csv", index=False)
    strata = stratum_auc(frame, "objective_rule_repeat_0")
    strata.to_csv(OUT / "stratified_auc_primary_split.csv", index=False)
    primary_threshold = float(repeat_results[(repeat_results["split"] == "objective_rule_repeat_0") & (repeat_results["rule"] == "centered_ncc")].iloc[0]["threshold"])
    conditions = centered_ncc_condition_performance(frame, "objective_rule_repeat_0", primary_threshold)
    conditions.to_csv(OUT / "centered_ncc_condition_performance_primary_split.csv", index=False)
    fixed = fixed_threshold_stability(frame, 0.160)
    fixed.to_csv(OUT / "centered_ncc_fixed_threshold_stability.csv", index=False)
    fixed_summary = pd.Series({
        "tpr_median": fixed["tpr"].median(),
        "far_median": fixed["far"].median(),
        "balanced_accuracy_median": fixed["balanced_accuracy"].median(),
    })
    (OUT / "summary.md").write_text(markdown_summary(frame, summary, strata, conditions, fixed_summary), encoding="utf-8")
    print((OUT / "summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
