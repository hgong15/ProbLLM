#!/usr/bin/env python
import argparse
import csv
import json
from pathlib import Path


DEFAULT_TARGETS = {
    ("CiteULike_item", "mf"): {"warm_recall@20": 0.3058, "warm_ndcg@20": 0.2061},
    ("ml-1m_item", "mf"): {"warm_recall@20": 0.2129, "warm_ndcg@20": 0.1924},
    ("CiteULike_item", "lgn"): {"warm_recall@20": 0.3252, "warm_ndcg@20": 0.2156},
    ("ml-1m_item", "lgn"): {"warm_recall@20": 0.3314, "warm_ndcg@20": 0.3186},
}


def read_metric(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    test = data.get("test", {})
    warm = test.get("warm", {})
    overall = test.get("overall", {})
    strict = test.get("strict_cold", {})
    return {
        "warm_recall@20": warm.get("recall@20", ""),
        "warm_ndcg@20": warm.get("ndcg@20", ""),
        "overall_recall@20": overall.get("recall@20", ""),
        "overall_ndcg@20": overall.get("ndcg@20", ""),
        "strict_cold_recall@20": strict.get("recall@20", ""),
        "strict_cold_ndcg@20": strict.get("ndcg@20", ""),
    }


def gap(row, target):
    recall = row.get("warm_recall@20")
    ndcg = row.get("warm_ndcg@20")
    if recall == "" or ndcg == "":
        return ""
    return abs(float(recall) - target["warm_recall@20"]) + abs(float(ndcg) - target["warm_ndcg@20"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--method", default="warm_only_e200_bs128_rerun20260623")
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 2020, 2021, 2022, 2023])
    args = parser.parse_args()

    out_dir = args.output_dir or args.root / "experiments" / "revision_main" / f"backbone_{args.method}_warm_target_selection"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    best = []
    for dataset_key, model in DEFAULT_TARGETS:
        target = DEFAULT_TARGETS[(dataset_key, model)]
        candidates = []
        for seed in args.seeds:
            metric_path = args.root / "experiments" / "revision_main" / dataset_key / model / args.method / f"seed_{seed}" / "final_metrics.json"
            row = {
                "dataset_key": dataset_key,
                "model": model,
                "seed": seed,
                "status": "ok" if metric_path.exists() else "missing",
                "target_warm_recall@20": target["warm_recall@20"],
                "target_warm_ndcg@20": target["warm_ndcg@20"],
                "source": str(metric_path),
            }
            if metric_path.exists():
                row.update(read_metric(metric_path))
                row["warm_target_l1_gap"] = gap(row, target)
                candidates.append(row)
            else:
                row["warm_target_l1_gap"] = ""
            rows.append(row)
        if candidates:
            best.append(min(candidates, key=lambda r: float(r["warm_target_l1_gap"])))

    fields = [
        "dataset_key",
        "model",
        "seed",
        "status",
        "target_warm_recall@20",
        "target_warm_ndcg@20",
        "warm_recall@20",
        "warm_ndcg@20",
        "warm_target_l1_gap",
        "overall_recall@20",
        "overall_ndcg@20",
        "strict_cold_recall@20",
        "strict_cold_ndcg@20",
        "source",
    ]
    for name, data in (("all_seed_warm_target_gap.csv", rows), ("best_seed_by_warm_target_gap.csv", best)):
        path = out_dir / name
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(data)
        print(f"saved={path}")


if __name__ == "__main__":
    main()
