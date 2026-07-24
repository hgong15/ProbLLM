import argparse
import csv
import json
import math
from pathlib import Path


DEFAULT_SEEDS = [42, 2020, 2021, 2022, 2023]
DEFAULT_SPLITS = ["overall", "strict_cold", "warmup", "warm"]
LATEX_SPLITS = ["overall", "strict_cold", "warmup"]
DEFAULT_METRICS = ["precision@20", "recall@20", "ndcg@20"]


def sample_std(values):
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))


def load_seed_metrics(experiment_root: Path, seed: int):
    path = experiment_root / f"seed_{seed}" / "final_metrics.json"
    if not path.exists():
        return []

    data = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for split, metrics in data.get("test", {}).items():
        for metric, value in metrics.items():
            rows.append(
                {
                    "seed": seed,
                    "split": split,
                    "metric": metric,
                    "value": float(value),
                    "source": str(path),
                }
            )
    return rows


def write_csv(path: Path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_summary(rows):
    grouped = {}
    for row in rows:
        key = (row["split"], row["metric"])
        grouped.setdefault(key, []).append(row)

    summary_rows = []
    for split in DEFAULT_SPLITS:
        for metric in DEFAULT_METRICS:
            values_rows = grouped.get((split, metric), [])
            values = [row["value"] for row in values_rows]
            if not values:
                continue

            mean = sum(values) / len(values)
            std = sample_std(values)
            seeds = " ".join(str(row["seed"]) for row in sorted(values_rows, key=lambda item: item["seed"]))
            summary_rows.append(
                {
                    "split": split,
                    "metric": metric,
                    "n": len(values),
                    "mean": f"{mean:.6f}",
                    "std": f"{std:.6f}",
                    "mean_std": f"{mean:.4f} ± {std:.4f}",
                    "seeds": seeds,
                }
            )
    return summary_rows


def write_latex(path: Path, summary_rows, method_name: str):
    by_split = {}
    for row in summary_rows:
        by_split.setdefault(row["split"], {})[row["metric"]] = row["mean_std"].replace(" ± ", r"$\pm$")

    lines = []
    for split in LATEX_SPLITS:
        metrics = by_split.get(split, {})
        if not metrics:
            continue
        lines.append(
            "{} ({}) & {} & {} & {} \\\\".format(
                method_name,
                split,
                metrics.get("precision@20", "-"),
                metrics.get("recall@20", "-"),
                metrics.get("ndcg@20", "-"),
            )
        )

    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def write_significance_placeholder(path: Path):
    rows = [
        {
            "status": "not_computed",
            "reason": "paired multi-seed comparator results were not provided",
            "expected_input": "paired comparator values for the same seeds",
        }
    ]
    write_csv(path, rows, ["status", "reason", "expected_input"])


def main():
    parser = argparse.ArgumentParser(description="Aggregate ProbLLM multi-seed final metrics.")
    parser.add_argument(
        "--experiment_root",
        default="./experiments/multiseed/CiteULike_item",
    )
    parser.add_argument("--method_name", default="Method")
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    args = parser.parse_args()

    experiment_root = Path(args.experiment_root)
    aggregate_dir = experiment_root / "aggregate"
    aggregate_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for seed in args.seeds:
        all_rows.extend(load_seed_metrics(experiment_root, seed))

    write_csv(
        aggregate_dir / "all_seed_metrics.csv",
        all_rows,
        ["seed", "split", "metric", "value", "source"],
    )

    summary_rows = build_summary(all_rows)
    write_csv(
        aggregate_dir / "summary_mean_std.csv",
        summary_rows,
        ["split", "metric", "n", "mean", "std", "mean_std", "seeds"],
    )
    write_latex(aggregate_dir / "latex_table_rows.tex", summary_rows, args.method_name)
    write_significance_placeholder(aggregate_dir / "significance_tests.csv")

    print(f"Aggregated {len(all_rows)} metric values from {len(args.seeds)} requested seeds.")
    print(f"Saved aggregate files to {aggregate_dir}")


if __name__ == "__main__":
    main()
