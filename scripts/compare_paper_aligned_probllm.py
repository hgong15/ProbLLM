#!/usr/bin/env python
import argparse
from pathlib import Path

import pandas as pd
from scipy import stats


METRIC_ORDER = ["precision@20", "recall@20", "ndcg@20"]
SPLIT_ORDER = ["overall", "strict_cold", "warmup", "warm"]


OLD_METHOD = {
    ("CiteULike_item", "mf"): "probllm_mf_fix",
    ("CiteULike_item", "lgn"): "probllm",
    ("ml-1m_item", "mf"): "probllm",
    ("ml-1m_item", "lgn"): "probllm",
}


def load_aggregate(root: Path, dataset_key: str, backbone: str, method: str):
    path = root / "experiments" / "revision_main" / dataset_key / backbone / method / "aggregate"
    summary_path = path / "summary_mean_std.csv"
    seeds_path = path / "all_seed_metrics.csv"
    if not summary_path.exists() or not seeds_path.exists():
        return None, None
    summary = pd.read_csv(summary_path)
    seeds = pd.read_csv(seeds_path)
    for df in (summary, seeds):
        df.insert(0, "dataset_key", dataset_key)
        df.insert(1, "backbone", backbone)
        df.insert(2, "method", method)
    return summary, seeds


def order_key(value, order):
    try:
        return order.index(value)
    except ValueError:
        return len(order)


def build_comparison(root: Path):
    summaries = []
    seeds = []
    for dataset_key in ("CiteULike_item", "ml-1m_item"):
        for backbone in ("mf", "lgn"):
            old_method = OLD_METHOD[(dataset_key, backbone)]
            for label, method in (("old", old_method), ("paper_aligned", "probllm_paper_aligned")):
                summary, seed = load_aggregate(root, dataset_key, backbone, method)
                if summary is None:
                    continue
                summary.insert(3, "version", label)
                seed.insert(3, "version", label)
                summaries.append(summary)
                seeds.append(seed)
    summary_df = pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame()
    seed_df = pd.concat(seeds, ignore_index=True) if seeds else pd.DataFrame()
    return summary_df, seed_df


def build_diff(summary_df: pd.DataFrame):
    rows = []
    for (dataset_key, backbone, split, metric), group in summary_df.groupby(
        ["dataset_key", "backbone", "split", "metric"]
    ):
        old = group[group["version"].eq("old")]
        new = group[group["version"].eq("paper_aligned")]
        if old.empty or new.empty:
            continue
        old_row = old.iloc[0]
        new_row = new.iloc[0]
        rows.append(
            {
                "dataset_key": dataset_key,
                "backbone": backbone,
                "split": split,
                "metric": metric,
                "old_method": old_row["method"],
                "new_method": new_row["method"],
                "old_mean": float(old_row["mean"]),
                "old_std": float(old_row["std"]),
                "new_mean": float(new_row["mean"]),
                "new_std": float(new_row["std"]),
                "delta_new_minus_old": float(new_row["mean"]) - float(old_row["mean"]),
                "relative_delta": (
                    (float(new_row["mean"]) - float(old_row["mean"])) / float(old_row["mean"])
                    if float(old_row["mean"]) != 0
                    else None
                ),
                "old_mean_std": old_row["mean_std"],
                "new_mean_std": new_row["mean_std"],
            }
        )
    diff = pd.DataFrame(rows)
    if diff.empty:
        return diff
    diff["_split_order"] = diff["split"].map(lambda x: order_key(x, SPLIT_ORDER))
    diff["_metric_order"] = diff["metric"].map(lambda x: order_key(x, METRIC_ORDER))
    return diff.sort_values(["dataset_key", "backbone", "_split_order", "_metric_order"]).drop(
        columns=["_split_order", "_metric_order"]
    )


def build_paired(seed_df: pd.DataFrame):
    rows = []
    for (dataset_key, backbone, split, metric), group in seed_df.groupby(
        ["dataset_key", "backbone", "split", "metric"]
    ):
        old = group[group["version"].eq("old")][["seed", "value"]]
        new = group[group["version"].eq("paper_aligned")][["seed", "value"]]
        merged = old.merge(new, on="seed", suffixes=("_old", "_new"))
        if merged.empty:
            continue
        if len(merged) >= 2:
            p_value = float(stats.ttest_rel(merged["value_new"], merged["value_old"]).pvalue)
        else:
            p_value = None
        rows.append(
            {
                "dataset_key": dataset_key,
                "backbone": backbone,
                "split": split,
                "metric": metric,
                "overlap_n": len(merged),
                "overlap_seeds": " ".join(str(int(seed)) for seed in sorted(merged["seed"].unique())),
                "old_mean_on_overlap": merged["value_old"].mean(),
                "new_mean_on_overlap": merged["value_new"].mean(),
                "delta_new_minus_old": merged["value_new"].mean() - merged["value_old"].mean(),
                "paired_t_p_value": p_value,
            }
        )
    paired = pd.DataFrame(rows)
    if paired.empty:
        return paired
    paired["_split_order"] = paired["split"].map(lambda x: order_key(x, SPLIT_ORDER))
    paired["_metric_order"] = paired["metric"].map(lambda x: order_key(x, METRIC_ORDER))
    return paired.sort_values(["dataset_key", "backbone", "_split_order", "_metric_order"]).drop(
        columns=["_split_order", "_metric_order"]
    )


def write_excel(output: Path, sheets):
    output.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        for name, df in sheets.items():
            df.to_excel(writer, sheet_name=name[:31], index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/revision_main/probllm_paper_aligned_comparison.xlsx"),
    )
    args = parser.parse_args()

    root = args.root.resolve()
    summary_df, seed_df = build_comparison(root)
    sheets = {
        "mean_std_all": summary_df,
        "mean_delta": build_diff(summary_df) if not summary_df.empty else pd.DataFrame(),
        "paired_tests": build_paired(seed_df) if not seed_df.empty else pd.DataFrame(),
        "per_seed_all": seed_df,
    }
    write_excel(root / args.output, sheets)
    print(f"Saved {root / args.output}")
    if not sheets["mean_delta"].empty:
        compact = sheets["mean_delta"][
            sheets["mean_delta"]["metric"].isin(["recall@20", "ndcg@20"])
            & sheets["mean_delta"]["split"].isin(["overall", "strict_cold", "warmup"])
        ]
        print(compact.to_string(index=False))


if __name__ == "__main__":
    main()
