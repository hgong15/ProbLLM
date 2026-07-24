#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.evaluate_simulation_quality import (
    DEFAULT_SEEDS,
    SPLIT_TO_GT,
    SPLIT_TO_VAL,
    evaluate_split,
    head_counterparts,
    load_edges,
    sample_std,
)


METRICS = [
    "targets",
    "scored_pairs",
    "positive_scored_pairs",
    "precision@20",
    "recall@20",
    "auc",
    "cal_brier",
    "cal_ece",
    "brier",
    "ece",
    "head@20",
]


def numeric(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def summarize(rows: list[dict], k: int) -> pd.DataFrame:
    metric_cols = [
        "targets",
        "scored_pairs",
        "positive_scored_pairs",
        f"precision@{k}",
        f"recall@{k}",
        "auc",
        "cal_brier",
        "cal_ece",
        "brier",
        "ece",
        f"head@{k}",
    ]
    out_rows = []
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    for (method, dataset, split), group in df.groupby(["method", "dataset", "split"], dropna=False):
        out = {"method": method, "dataset": dataset, "split": split, "n": int(len(group))}
        for metric in metric_cols:
            values = [float(value) for value in group[metric].tolist() if not pd.isna(value)]
            if not values:
                continue
            mean = sum(values) / len(values)
            std = sample_std(values)
            out[f"{metric}_mean"] = mean
            out[f"{metric}_std"] = std
            out[f"{metric}_mean_std"] = f"{mean:.4f} ± {std:.4f}"
        out_rows.append(out)
    return pd.DataFrame(out_rows)


def candidate_methods(root: Path, dataset_key: str, model: str, seeds: list[int]):
    base = root / "experiments" / "revision_main" / dataset_key / model
    methods: dict[str, dict[int, Path]] = {}
    for pred_path in base.glob("*/seed_*/predicted_cold_item_interaction.csv"):
        method = pred_path.parents[1].name
        seed_name = pred_path.parent.name
        if not seed_name.startswith("seed_"):
            continue
        try:
            seed = int(seed_name.removeprefix("seed_"))
        except ValueError:
            continue
        if seed in seeds:
            methods.setdefault(method, {})[seed] = pred_path.parent
    for method, seed_dirs in sorted(methods.items()):
        if all(seed in seed_dirs for seed in seeds):
            yield method, seed_dirs


def required_artifacts(seed_dir: Path, splits: list[str]) -> bool:
    if not (seed_dir / "predicted_cold_item_interaction.csv").exists():
        return False
    for name in ["warm_train.csv", "warmup_support.csv"]:
        if not (seed_dir / name).exists():
            return False
    for split in splits:
        for name in [SPLIT_TO_GT[split], SPLIT_TO_VAL[split]]:
            if not (seed_dir / name).exists():
                return False
    return True


def evaluate_method(method: str, seed_dirs: dict[int, Path], dataset_key: str, seeds: list[int], splits: list[str], k: int):
    rows = []
    for seed in seeds:
        seed_dir = seed_dirs[seed]
        pred = pd.read_csv(seed_dir / "predicted_cold_item_interaction.csv")
        pred["probability"] = pd.to_numeric(pred.get("probability", 1.0), errors="coerce").fillna(1.0)
        head_users = head_counterparts(
            [seed_dir / "warm_train.csv", seed_dir / "warmup_support.csv"],
            counterpart_col="user",
        )
        for split_name in splits:
            positives = load_edges(seed_dir / SPLIT_TO_GT[split_name])
            val_positives = load_edges(seed_dir / SPLIT_TO_VAL[split_name])
            result = evaluate_split(pred, positives, split_name, head_users, k, val_positives=val_positives)
            if result is None:
                continue
            result.update({"method": method, "dataset": dataset_key, "seed": seed, "seed_dir": str(seed_dir)})
            rows.append(result)
    return rows


def load_reference(path: Path, dataset_key: str, splits: list[str], k: int):
    if not path.exists():
        return {}
    ref = pd.read_csv(path)
    out = {}
    for row in ref.itertuples(index=False):
        row_dict = row._asdict()
        if row_dict.get("dataset") != dataset_key or row_dict.get("split") not in splits:
            continue
        split = row_dict["split"]
        out[split] = {
            f"precision@{k}": numeric(row_dict.get(f"precision@{k}_mean")),
            f"recall@{k}": numeric(row_dict.get(f"recall@{k}_mean")),
            "auc": numeric(row_dict.get("auc_mean")),
            "brier": numeric(row_dict.get("brier_mean")),
            "ece": numeric(row_dict.get("ece_mean")),
            f"head@{k}": numeric(row_dict.get(f"head@{k}_mean")),
        }
    return out


def match_scores(summary: pd.DataFrame, reference: dict, k: int):
    if not reference or summary.empty:
        return pd.DataFrame()
    rows = []
    metric_names = [f"precision@{k}", f"recall@{k}", "auc", "brier", "ece", f"head@{k}"]
    for method, group in summary.groupby("method"):
        diff = 0.0
        terms = 0
        for row in group.itertuples(index=False):
            row_dict = row._asdict()
            split = row_dict["split"]
            if split not in reference:
                continue
            for metric in metric_names:
                ref_value = reference[split].get(metric, float("nan"))
                got_value = numeric(row_dict.get(f"{metric}_mean"))
                if math.isnan(ref_value) or math.isnan(got_value):
                    continue
                diff += abs(got_value - ref_value)
                terms += 1
        rows.append({"method": method, "match_l1": diff / max(terms, 1), "matched_terms": terms})
    return pd.DataFrame(rows).sort_values(["match_l1", "method"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan CiteULike simulated-interaction files and add validation-bin calibration metrics.")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--dataset_key", default="CiteULike_item")
    parser.add_argument("--model", default="lgn")
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--splits", nargs="+", default=["strict_cold", "warmup"])
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--reference_summary", type=Path, default=None)
    parser.add_argument("--out_dir", type=Path, default=None)
    parser.add_argument("--limit_methods", type=int, default=0)
    args = parser.parse_args()

    root = args.root
    out_dir = args.out_dir or root / "experiments" / "revision_diagnostics" / "simulation_quality_calibrated_scan_20260701"
    if not out_dir.is_absolute():
        out_dir = root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    reference_path = args.reference_summary or root / "experiments" / "revision_diagnostics" / "simulation_quality" / "summary_mean_std.csv"
    reference = load_reference(reference_path, args.dataset_key, args.splits, args.k)

    all_rows = []
    skipped = []
    method_iter = list(candidate_methods(root, args.dataset_key, args.model, args.seeds))
    if args.limit_methods > 0:
        method_iter = method_iter[: args.limit_methods]
    for idx, (method, seed_dirs) in enumerate(method_iter, start=1):
        missing = [
            f"seed_{seed}"
            for seed in args.seeds
            if not required_artifacts(seed_dirs[seed], args.splits)
        ]
        if missing:
            skipped.append({"method": method, "reason": "missing_artifacts", "detail": ",".join(missing)})
            continue
        print(f"[{idx}/{len(method_iter)}] evaluate {method}", flush=True)
        try:
            all_rows.extend(evaluate_method(method, seed_dirs, args.dataset_key, args.seeds, args.splits, args.k))
        except Exception as exc:
            skipped.append({"method": method, "reason": type(exc).__name__, "detail": str(exc)})

    all_df = pd.DataFrame(all_rows)
    all_df.to_csv(out_dir / "all_method_seed_metrics.csv", index=False)
    if skipped:
        pd.DataFrame(skipped).to_csv(out_dir / "skipped_methods.csv", index=False)

    summary = summarize(all_rows, args.k)
    summary.to_csv(out_dir / "method_summary_mean_std.csv", index=False)
    scores = match_scores(summary, reference, args.k)
    if not scores.empty:
        scores.to_csv(out_dir / "method_match_scores.csv", index=False)
        best_method = str(scores.iloc[0]["method"])
        summary[summary["method"] == best_method].to_csv(out_dir / "best_match_summary_mean_std.csv", index=False)
        all_df[all_df["method"] == best_method].to_csv(out_dir / "best_match_all_seed_metrics.csv", index=False)
        print("Best raw-metric match:", best_method, flush=True)
        print(summary[summary["method"] == best_method].to_string(index=False), flush=True)
    else:
        print("No reference match score computed; wrote all summaries.", flush=True)
    print(f"Saved calibration scan outputs to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
