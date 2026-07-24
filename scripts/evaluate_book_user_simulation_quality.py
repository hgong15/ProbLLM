#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd


DEFAULT_SEEDS = [42, 2020, 2021, 2022, 2023]
SPLIT_TO_GT = {
    "strict_cold": "cold_user_test.csv",
    "warmup": "warmup_test.csv",
}
SPLIT_TO_VAL = {
    "strict_cold": "cold_user_val.csv",
    "warmup": "warmup_val.csv",
}


def sample_std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))


def auc_from_labels(scores: list[float], labels: list[int]) -> float:
    if not labels:
        return float("nan")
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranked = pd.Series(scores).rank(method="average", ascending=True)
    rank_sum_pos = float(ranked[pd.Series(labels).astype(bool)].sum())
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def ece_score(scores: list[float], labels: list[int], bins: int = 10) -> float:
    if not labels:
        return float("nan")
    total = len(labels)
    ece = 0.0
    for bin_idx in range(bins):
        lo = bin_idx / bins
        hi = (bin_idx + 1) / bins
        in_bin = [
            idx
            for idx, score in enumerate(scores)
            if score >= lo and (score < hi or (bin_idx == bins - 1 and score <= hi))
        ]
        if not in_bin:
            continue
        acc = sum(labels[idx] for idx in in_bin) / len(in_bin)
        conf = sum(scores[idx] for idx in in_bin) / len(in_bin)
        ece += len(in_bin) / total * abs(acc - conf)
    return ece


def fit_bin_calibrator(scores: list[float], labels: list[int], bins: int = 10) -> list[float]:
    if not labels:
        return [0.0] * bins
    global_rate = sum(labels) / len(labels)
    values = []
    for bin_idx in range(bins):
        lo = bin_idx / bins
        hi = (bin_idx + 1) / bins
        in_bin = [
            idx
            for idx, score in enumerate(scores)
            if score >= lo and (score < hi or (bin_idx == bins - 1 and score <= hi))
        ]
        values.append(sum(labels[idx] for idx in in_bin) / len(in_bin) if in_bin else global_rate)
    return values


def apply_bin_calibrator(scores: list[float], bin_values: list[float]) -> list[float]:
    bins = len(bin_values)
    calibrated = []
    for score in scores:
        idx = min(max(int(score * bins), 0), bins - 1)
        calibrated.append(float(bin_values[idx]))
    return calibrated


def load_edges(path: Path) -> set[tuple[int, int]]:
    df = pd.read_csv(path, usecols=["user", "item"])
    return {(int(user), int(item)) for user, item in zip(df["user"], df["item"])}


def head_items(data_dir: Path, quantile: float = 0.8) -> set[int]:
    pieces = []
    for name in ("warm_emb.csv", "warm_train.csv", "warmup_support.csv"):
        path = data_dir / name
        if path.exists():
            pieces.append(pd.read_csv(path, usecols=["item"]))
    if not pieces:
        return set()
    train = pd.concat(pieces, ignore_index=True)
    counts = train["item"].astype(int).value_counts()
    if counts.empty:
        return set()
    threshold = counts.quantile(quantile)
    return set(int(item) for item, value in counts.items() if value >= threshold)


def label_pairs(df: pd.DataFrame, positives: set[tuple[int, int]]) -> list[int]:
    return [1 if (int(user), int(item)) in positives else 0 for user, item in zip(df["user"], df["item"])]


def evaluate_split(
    pred: pd.DataFrame,
    positives: set[tuple[int, int]],
    split_name: str,
    head_item_set: set[int],
    k: int,
    val_positives: set[tuple[int, int]] | None = None,
) -> dict[str, float | int | str] | None:
    split_pred = pred[pred["entity_type"].eq(split_name)].copy()
    if split_pred.empty:
        return None

    split_pred["label"] = label_pairs(split_pred, positives)
    split_pred["probability"] = pd.to_numeric(split_pred["probability"], errors="coerce").fillna(0.0).clip(0.0, 1.0)

    positives_by_user: dict[int, set[int]] = {}
    for user, item in positives:
        positives_by_user.setdefault(int(user), set()).add(int(item))

    top = (
        split_pred.sort_values(["user", "probability"], ascending=[True, False])
        .groupby("user", sort=False)
        .head(k)
        .copy()
    )
    top["top_hit"] = label_pairs(top, positives)
    top["is_head"] = [1 if int(item) in head_item_set else 0 for item in top["item"]]

    precision_values = []
    recall_values = []
    head_values = []
    for user, gt_items in positives_by_user.items():
        group = top[top["user"].astype(int).eq(user)]
        if group.empty:
            continue
        hits = int(group["top_hit"].sum())
        precision_values.append(hits / max(len(group), 1))
        recall_values.append(hits / len(gt_items))
        head_values.append(float(group["is_head"].mean()))

    scores = [float(value) for value in split_pred["probability"].tolist()]
    labels = [int(value) for value in split_pred["label"].tolist()]
    brier = sum((score - label) ** 2 for score, label in zip(scores, labels)) / len(labels)
    raw_ece = ece_score(scores, labels)
    cal_brier = float("nan")
    cal_ece = float("nan")
    if val_positives is not None:
        val_labels = label_pairs(split_pred, val_positives)
        bin_values = fit_bin_calibrator(scores, val_labels)
        cal_scores = apply_bin_calibrator(scores, bin_values)
        cal_brier = sum((score - label) ** 2 for score, label in zip(cal_scores, labels)) / len(labels)
        cal_ece = ece_score(cal_scores, labels)

    return {
        "split": split_name,
        "targets": len(precision_values),
        "scored_pairs": len(labels),
        "positive_scored_pairs": sum(labels),
        f"precision@{k}": sum(precision_values) / len(precision_values) if precision_values else float("nan"),
        f"recall@{k}": sum(recall_values) / len(recall_values) if recall_values else float("nan"),
        "auc": auc_from_labels(scores, labels),
        "cal_brier": cal_brier,
        "cal_ece": cal_ece,
        "brier": brier,
        "ece": raw_ece,
        f"head@{k}": sum(head_values) / len(head_values) if head_values else float("nan"),
    }


def data_dir_for_seed(root: Path, seed: int) -> Path:
    preferred = root / "data" / f"book-crossing_userllm_seed{seed}"
    if preferred.exists():
        return preferred
    fallback = root / "data" / "book-crossing"
    if fallback.exists():
        return fallback
    raise FileNotFoundError(preferred)


def evaluate_seed(root: Path, llm_root: Path, method: str, seed: int, k: int) -> list[dict[str, object]]:
    data_dir = data_dir_for_seed(root, seed)
    pred_csv = llm_root / method / f"seed_{seed}" / "predicted_cold_user_interaction_all.csv"
    if not pred_csv.exists():
        raise FileNotFoundError(pred_csv)

    pred = pd.read_csv(pred_csv)
    required = {"user", "item", "entity_type", "probability"}
    missing = required - set(pred.columns)
    if missing:
        raise ValueError(f"{pred_csv} missing columns: {sorted(missing)}")
    pred = pred[["user", "item", "entity_type", "probability"]].copy()
    pred["user"] = pred["user"].astype(int)
    pred["item"] = pred["item"].astype(int)
    pred["entity_type"] = pred["entity_type"].astype(str)

    heads = head_items(data_dir)
    rows = []
    for split_name, gt_file in SPLIT_TO_GT.items():
        positives = load_edges(data_dir / gt_file)
        val_positives = None
        val_file = SPLIT_TO_VAL.get(split_name)
        if val_file is not None and (data_dir / val_file).exists():
            val_positives = load_edges(data_dir / val_file)
        result = evaluate_split(pred, positives, split_name, heads, k, val_positives=val_positives)
        if result is None:
            continue
        result.update({"dataset": "Book-Crossing user", "seed": seed, "data_dir": str(data_dir), "pred_csv": str(pred_csv)})
        rows.append(result)
    return rows


def write_summary(out_dir: Path, rows: list[dict[str, object]], k: int) -> None:
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
    df = pd.DataFrame(rows)
    summary_rows = []
    for (dataset, split), group in df.groupby(["dataset", "split"]):
        out: dict[str, object] = {"dataset": dataset, "split": split, "n": len(group)}
        for metric in metric_cols:
            values = [float(value) for value in group[metric].tolist() if not pd.isna(value)]
            if not values:
                continue
            mean = sum(values) / len(values)
            std = sample_std(values)
            out[f"{metric}_mean"] = mean
            out[f"{metric}_std"] = std
            out[f"{metric}_mean_std"] = f"{mean:.4f} ({std:.4f})"
        summary_rows.append(out)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / "summary_mean_std.csv", index=False)
    with (out_dir / "latex_rows.txt").open("w", encoding="utf-8") as handle:
        for _, row in summary.iterrows():
            split = str(row["split"]).replace("_", " ").title().replace("Cold", "Cold-Start")
            handle.write(
                "Book-Crossing user & "
                f"{split} & "
                f"\\meansd{{{row[f'precision@{k}_mean']:.4f}}}{{{row[f'precision@{k}_std']:.4f}}} & "
                f"\\meansd{{{row[f'recall@{k}_mean']:.4f}}}{{{row[f'recall@{k}_std']:.4f}}} & "
                f"\\meansd{{{row['auc_mean']:.4f}}}{{{row['auc_std']:.4f}}} & "
                f"\\meansd{{{row['cal_brier_mean']:.4f}}}{{{row['cal_brier_std']:.4f}}} & "
                f"\\meansd{{{row['cal_ece_mean']:.4f}}}{{{row['cal_ece_std']:.4f}}} & "
                f"\\meansd{{{row['brier_mean']:.4f}}}{{{row['brier_std']:.4f}}} & "
                f"\\meansd{{{row['ece_mean']:.4f}}}{{{row['ece_std']:.4f}}} & "
                f"\\meansd{{{row[f'head@{k}_mean']:.4f}}}{{{row[f'head@{k}_std']:.4f}}} \\\\\n"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Book-Crossing user-side simulated interaction quality.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--llm_root", default=None)
    parser.add_argument("--method", default="probllm_paper_aligned_7b_content_neighbor_top50")
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--out_dir", default="experiments/revision_diagnostics/book_user_simulation_quality")
    args = parser.parse_args()

    root = Path(args.root)
    llm_root = Path(args.llm_root) if args.llm_root else root / "experiments" / "llm_outputs" / "book-crossing_user_content_neighbor_top50_select20_k100"
    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for seed in args.seeds:
        rows.extend(evaluate_seed(root, llm_root, args.method, seed, args.k))

    pd.DataFrame(rows).to_csv(out_dir / "all_seed_metrics.csv", index=False)
    write_summary(out_dir, rows, args.k)
    print(f"Saved Book-Crossing user simulation-quality diagnostics to {out_dir}")


if __name__ == "__main__":
    main()
