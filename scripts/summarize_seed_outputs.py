import argparse
import csv
import json
import os
import re
from pathlib import Path


def count_csv(path: Path):
    if not path.exists():
        return {"exists": False, "records": 0, "users": 0, "items": 0}

    users = set()
    items = set()
    records = 0
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records += 1
            if "user" in row:
                users.add(row["user"])
            if "item" in row:
                items.add(row["item"])

    return {
        "exists": True,
        "records": records,
        "users": len(users),
        "items": len(items),
    }


def parse_metric_line(line: str):
    metric = {}
    for name in ("precision", "recall", "ndcg"):
        match = re.search(rf"'{name}': array\(\[([^\]]+)\]\)", line)
        if not match:
            continue
        values = [float(value.strip()) for value in match.group(1).split(",") if value.strip()]
        if values:
            metric[f"{name}@20"] = values[0]
    return metric


def parse_split_metrics(lines):
    metrics = {}
    for line in lines:
        if "Strict Cold-Start Result:" in line:
            metrics["strict_cold"] = parse_metric_line(line)
        elif "Cold Result:" in line:
            metrics["strict_cold"] = parse_metric_line(line)
        elif "Warm-Up Result:" in line:
            metrics["warmup"] = parse_metric_line(line)
        elif "Warm Result:" in line:
            metrics["warm"] = parse_metric_line(line)
        elif "Overall Result:" in line:
            metrics["overall"] = parse_metric_line(line)
    return metrics


def parse_final_metrics(log_path: Path, seed: int):
    metrics = {"seed": seed, "val": {}, "test": {}, "source": str(log_path)}
    if not log_path.exists():
        metrics["warning"] = "log file not found"
        return metrics

    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    val_start = None
    test_start = 0
    for idx, line in enumerate(lines):
        if "[VAL]" in line:
            val_start = idx
        if "[TEST]" in line:
            test_start = idx

    if val_start is not None and val_start < test_start:
        metrics["val"] = parse_split_metrics(lines[val_start:test_start])
    metrics["test"] = parse_split_metrics(lines[test_start:])

    if not metrics["test"]:
        metrics["warning"] = "no final test metrics parsed from log"
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Summarize one multi-seed ProbLLM run.")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--seed_dir", required=True)
    parser.add_argument("--log_file", required=True)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    seed_dir = Path(args.seed_dir)
    seed_dir.mkdir(parents=True, exist_ok=True)

    split_files = [
        "warm_emb.csv",
        "warm_train.csv",
        "warm_emb_original.csv",
        "warm_val.csv",
        "warm_test.csv",
        "warmup_support.csv",
        "warmup_val.csv",
        "warmup_test.csv",
        "cold_item.csv",
        "cold_item_val.csv",
        "cold_item_test.csv",
        "cold_user.csv",
        "cold_user_val.csv",
        "cold_user_test.csv",
        "overall_val.csv",
        "overall_test.csv",
        "top20.csv",
        "predicted_cold_item_interaction.csv",
    ]
    split_summary = {
        "seed": args.seed,
        "data_dir": str(data_dir),
        "files": {name: count_csv(data_dir / name) for name in split_files},
    }
    (seed_dir / "split_summary.json").write_text(
        json.dumps(split_summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    final_metrics = parse_final_metrics(Path(args.log_file), args.seed)
    (seed_dir / "final_metrics.json").write_text(
        json.dumps(final_metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    for name in ("top20.csv", "predicted_cold_item_interaction.csv", "split_meta.json"):
        src = data_dir / name
        if src.exists():
            dst = seed_dir / name
            dst.write_bytes(src.read_bytes())

    print(f"Saved summaries to {seed_dir}")


if __name__ == "__main__":
    main()
