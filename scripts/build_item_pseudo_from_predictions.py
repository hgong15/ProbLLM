import argparse
import csv
import json
import math
import re
from pathlib import Path

import pandas as pd


def parse_probability(value):
    if value is None:
        return None
    if isinstance(value, (int, float)):
        prob = float(value)
    else:
        text = str(value).strip()
        try:
            prob = float(text)
        except ValueError:
            matches = re.findall(r"(?<![\d.])(?:0(?:\.\d+)?|1(?:\.0+)?)(?!\d)", text)
            if not matches:
                return None
            prob = float(matches[0])
    if math.isnan(prob) or prob < 0.0 or prob > 1.0:
        return None
    return prob


def read_predictions(path):
    predictions = []
    invalid = 0
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                predictions.append(None)
                invalid += 1
                continue
            prob = parse_probability(record.get("predict"))
            if prob is None:
                invalid += 1
            predictions.append(prob)
    return predictions, invalid


def main():
    parser = argparse.ArgumentParser(description="Build item-cold pseudo interactions from LLM prediction JSONL.")
    parser.add_argument("--top_csv", required=True)
    parser.add_argument("--prediction_jsonl", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--topn_per_item", type=int, default=0, help="0 keeps every row above threshold.")
    parser.add_argument("--min_per_item", type=int, default=0, help="Backfill highest scored rows up to this count.")
    parser.add_argument(
        "--output_probability",
        choices=["raw", "constant", "item_rank"],
        default="raw",
        help="How to write selected pseudo-edge probabilities.",
    )
    parser.add_argument("--constant_probability", type=float, default=1.0)
    parser.add_argument("--rank_min_probability", type=float, default=0.6)
    parser.add_argument("--rank_max_probability", type=float, default=1.0)
    parser.add_argument("--summary_json", default=None)
    args = parser.parse_args()

    candidates = pd.read_csv(args.top_csv)
    predictions, invalid = read_predictions(args.prediction_jsonl)
    if len(candidates) != len(predictions):
        raise ValueError(
            f"Length mismatch: candidates={len(candidates)} predictions={len(predictions)}"
        )

    rows = []
    for row, prob in zip(candidates.to_dict("records"), predictions):
        if prob is None:
            continue
        record = {
            "user": int(row["user"]),
            "item": int(row["item"]),
            "entity_type": row.get("entity_type", ""),
            "probability": float(prob),
        }
        record["_below_threshold"] = prob < args.threshold
        rows.append(record)

    selected = []
    for (_, entity_type), group in pd.DataFrame(rows).groupby(["item", "entity_type"], dropna=False):
        records = group.to_dict("records")
        records.sort(key=lambda r: (-float(r["probability"]), int(r["user"])))
        above = [r for r in records if not r["_below_threshold"]]
        if args.min_per_item:
            keep = above[:]
            seen = {(r["user"], r["item"], r["entity_type"]) for r in keep}
            for record in records:
                key = (record["user"], record["item"], record["entity_type"])
                if key in seen:
                    continue
                keep.append(record)
                seen.add(key)
                if len(keep) >= args.min_per_item:
                    break
        else:
            keep = above
        if args.topn_per_item:
            keep = keep[: args.topn_per_item]
        if args.output_probability == "constant":
            for record in keep:
                record["probability"] = args.constant_probability
        elif args.output_probability == "item_rank" and keep:
            denom = max(len(keep) - 1, 1)
            for idx, record in enumerate(keep):
                frac = 1.0 - idx / denom
                record["probability"] = args.rank_min_probability + (
                    args.rank_max_probability - args.rank_min_probability
                ) * frac
        selected.extend(keep)

    selected.sort(key=lambda r: (int(r["item"]), str(r["entity_type"]), -float(r["probability"]), int(r["user"])))
    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["user", "item", "entity_type", "probability"])
        writer.writeheader()
        for row in selected:
            writer.writerow(
                {
                    "user": row["user"],
                    "item": row["item"],
                    "entity_type": row["entity_type"],
                    "probability": f"{float(row['probability']):.6f}",
                }
            )

    probs = [row["probability"] for row in rows if row["probability"] is not None]
    summary = {
        "top_csv": str(Path(args.top_csv).resolve()),
        "prediction_jsonl": str(Path(args.prediction_jsonl).resolve()),
        "output_csv": str(out_path.resolve()),
        "threshold": args.threshold,
        "topn_per_item": args.topn_per_item,
        "min_per_item": args.min_per_item,
        "output_probability": args.output_probability,
        "constant_probability": args.constant_probability,
        "rank_min_probability": args.rank_min_probability,
        "rank_max_probability": args.rank_max_probability,
        "candidate_rows": int(len(candidates)),
        "prediction_rows": int(len(predictions)),
        "invalid_predictions": int(invalid),
        "selected_rows": int(len(selected)),
        "probability": {
            "min": float(min(probs)) if probs else None,
            "max": float(max(probs)) if probs else None,
            "mean": float(sum(probs) / len(probs)) if probs else None,
        },
    }
    text = json.dumps(summary, indent=2, ensure_ascii=False)
    print(text)
    if args.summary_json:
        Path(args.summary_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.summary_json).write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
