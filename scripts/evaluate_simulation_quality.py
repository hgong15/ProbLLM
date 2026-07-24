#!/usr/bin/env python3
import argparse
import math
from pathlib import Path

import pandas as pd


DEFAULT_SEEDS = [42, 2020, 2021, 2022, 2023]
SPLIT_TO_GT = {
    "strict_cold": "cold_item_test.csv",
    "warmup": "warmup_test.csv",
}
SPLIT_TO_VAL = {
    "strict_cold": "cold_item_val.csv",
    "warmup": "warmup_val.csv",
}


def sample_std(values):
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))


def auc_from_labels(scores, labels):
    n = len(labels)
    n_pos = sum(labels)
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = sorted(range(n), key=lambda idx: scores[idx])
    rank_sum_pos = 0.0
    i = 0
    while i < n:
        j = i + 1
        while j < n and scores[order[j]] == scores[order[i]]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for pos in range(i, j):
            if labels[order[pos]]:
                rank_sum_pos += avg_rank
        i = j
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def ece_score(scores, labels, bins=10):
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
            if (score >= lo and (score < hi or (bin_idx == bins - 1 and score <= hi)))
        ]
        if not in_bin:
            continue
        acc = sum(labels[idx] for idx in in_bin) / len(in_bin)
        conf = sum(scores[idx] for idx in in_bin) / len(in_bin)
        ece += len(in_bin) / total * abs(acc - conf)
    return ece


def fit_bin_calibrator(scores, labels, bins=10):
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
            if (score >= lo and (score < hi or (bin_idx == bins - 1 and score <= hi)))
        ]
        if in_bin:
            values.append(sum(labels[idx] for idx in in_bin) / len(in_bin))
        else:
            values.append(global_rate)
    return values


def apply_bin_calibrator(scores, bin_values):
    bins = len(bin_values)
    calibrated = []
    for score in scores:
        idx = min(max(int(score * bins), 0), bins - 1)
        calibrated.append(float(bin_values[idx]))
    return calibrated


def load_edges(path):
    df = pd.read_csv(path)
    return {(int(row.user), int(row.item)) for row in df.itertuples(index=False)}


def head_counterparts(train_paths, counterpart_col, quantile=0.8):
    pieces = []
    for path in train_paths:
        if path.exists():
            pieces.append(pd.read_csv(path))
    if not pieces:
        return set()
    train = pd.concat(pieces, ignore_index=True)
    counts = train[counterpart_col].value_counts()
    if counts.empty:
        return set()
    threshold = counts.quantile(quantile)
    return set(int(idx) for idx, value in counts.items() if value >= threshold)


def evaluate_split(pred, positives, split_name, head_users, k, val_positives=None):
    split_pred = pred[pred["entity_type"].eq(split_name)].copy()
    if split_pred.empty:
        return None

    split_pred["label"] = [
        1 if (int(row.user), int(row.item)) in positives else 0
        for row in split_pred.itertuples(index=False)
    ]
    split_pred["probability"] = split_pred["probability"].clip(0.0, 1.0)

    precision_values = []
    recall_values = []
    head_values = []
    grouped = split_pred.sort_values(["item", "probability"], ascending=[True, False]).groupby("item")
    positives_by_item = {}
    for user, item in positives:
        positives_by_item.setdefault(item, set()).add(user)

    for item, group in grouped:
        item = int(item)
        gt_users = positives_by_item.get(item, set())
        if not gt_users:
            continue
        top = group.head(k)
        pred_users = [int(user) for user in top["user"].tolist()]
        hits = sum(1 for user in pred_users if user in gt_users)
        precision_values.append(hits / max(len(pred_users), 1))
        recall_values.append(hits / len(gt_users))
        head_values.append(sum(1 for user in pred_users if user in head_users) / max(len(pred_users), 1))

    scores = [float(value) for value in split_pred["probability"].tolist()]
    labels = [int(value) for value in split_pred["label"].tolist()]
    brier = sum((score - label) ** 2 for score, label in zip(scores, labels)) / len(labels)
    cal_brier = float("nan")
    cal_ece = float("nan")
    if val_positives is not None:
        val_labels = [
            1 if (int(row.user), int(row.item)) in val_positives else 0
            for row in split_pred.itertuples(index=False)
        ]
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
        "ece": ece_score(scores, labels),
        f"head@{k}": sum(head_values) / len(head_values) if head_values else float("nan"),
    }


def evaluate_seed(root, dataset_key, seed, k):
    seed_pred = root / "experiments" / "multiseed" / dataset_key / f"seed_{seed}" / "predicted_cold_item_interaction.csv"
    seed_artifacts = (
        root
        / "experiments"
        / "revision_main"
        / dataset_key
        / "lgn"
        / "probllm_paper_aligned"
        / f"seed_{seed}"
    )
    if not seed_pred.exists():
        raise FileNotFoundError(seed_pred)
    if not seed_artifacts.exists():
        raise FileNotFoundError(seed_artifacts)

    pred = pd.read_csv(seed_pred)
    pred["probability"] = pd.to_numeric(pred.get("probability", 1.0), errors="coerce").fillna(1.0)
    head_users = head_counterparts(
        [seed_artifacts / "warm_train.csv", seed_artifacts / "warmup_support.csv"],
        counterpart_col="user",
    )

    rows = []
    for split_name, gt_file in SPLIT_TO_GT.items():
        positives = load_edges(seed_artifacts / gt_file)
        val_positives = None
        val_file = SPLIT_TO_VAL.get(split_name)
        if val_file is not None and (seed_artifacts / val_file).exists():
            val_positives = load_edges(seed_artifacts / val_file)
        result = evaluate_split(pred, positives, split_name, head_users, k, val_positives=val_positives)
        if result is None:
            continue
        result.update({"dataset": dataset_key, "seed": seed})
        rows.append(result)
    return rows


def write_summary(out_dir, rows, k):
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
    summary_rows = []
    for (dataset, split), group in pd.DataFrame(rows).groupby(["dataset", "split"]):
        out = {"dataset": dataset, "split": split, "n": len(group)}
        for metric in metric_cols:
            values = [float(value) for value in group[metric].tolist() if not pd.isna(value)]
            if not values:
                continue
            mean = sum(values) / len(values)
            std = sample_std(values)
            out[f"{metric}_mean"] = f"{mean:.6f}"
            out[f"{metric}_std"] = f"{std:.6f}"
            out[f"{metric}_mean_std"] = f"{mean:.4f} ± {std:.4f}"
        summary_rows.append(out)

    pd.DataFrame(summary_rows).to_csv(out_dir / "summary_mean_std.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description="Evaluate simulated interaction quality against held-out edges.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--datasets", nargs="+", default=["CiteULike_item", "ml-1m_item"])
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--out_dir", default="experiments/revision_diagnostics/simulation_quality")
    args = parser.parse_args()

    root = Path(args.root)
    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for dataset in args.datasets:
        for seed in args.seeds:
            rows.extend(evaluate_seed(root, dataset, seed, args.k))

    pd.DataFrame(rows).to_csv(out_dir / "all_seed_metrics.csv", index=False)
    write_summary(out_dir, rows, args.k)
    print(f"Saved simulation-quality diagnostics to {out_dir}")


if __name__ == "__main__":
    main()
