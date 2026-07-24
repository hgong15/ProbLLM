#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter MovieLens cold-user LLM pseudo interactions by probability and per-user quota."
    )
    parser.add_argument("--jsonl_path", type=Path, required=True)
    parser.add_argument("--top20_csv", type=Path, required=True)
    parser.add_argument("--output_csv", type=Path, required=True)
    parser.add_argument("--diagnostics_json", type=Path, default=None)
    parser.add_argument("--data_dir", type=Path, default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--topn_per_user", type=int, default=0, help="0 keeps all rows above threshold.")
    parser.add_argument("--min_per_user", type=int, default=0, help="Backfill top predictions up to this count per user.")
    parser.add_argument("--max_per_item", type=int, default=0, help="0 disables global item cap.")
    parser.add_argument("--pop_penalty", type=float, default=0.0, help="Rank by prob - alpha * normalized log item popularity.")
    parser.add_argument("--entity_type", choices=["all", "strict_cold", "warmup"], default="all")
    return parser.parse_args()


def parse_probability(value) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        prob = float(value)
    else:
        text = str(value).strip()
        try:
            prob = float(text)
        except ValueError:
            matches = re.findall(r"(?<!\d)(?:0(?:\.\d+)?|1(?:\.0+)?)(?!\d)", text)
            if not matches:
                return None
            prob = float(matches[0])
    if prob < 0 or prob > 1:
        return None
    return prob


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_predictions(path: Path) -> tuple[list[float | None], int]:
    predictions: list[float | None] = []
    invalid = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                value = json.loads(line).get("predict")
            except json.JSONDecodeError:
                value = None
            prob = parse_probability(value)
            if prob is None:
                invalid += 1
            predictions.append(prob)
    return predictions, invalid


def read_item_popularity(data_dir: Path | None) -> dict[int, float]:
    if data_dir is None:
        return {}
    path = data_dir / "warm_emb.csv"
    if not path.exists():
        return {}
    counts: Counter[int] = Counter()
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            counts[int(row["item"])] += 1
    if not counts:
        return {}
    import math

    values = {item: math.log1p(count) for item, count in counts.items()}
    max_value = max(values.values()) or 1.0
    return {item: value / max_value for item, value in values.items()}


def main() -> None:
    args = parse_args()
    candidates = read_csv_rows(args.top20_csv)
    predictions, invalid = read_predictions(args.jsonl_path)
    popularity = read_item_popularity(args.data_dir)
    if len(candidates) != len(predictions):
        raise ValueError(
            f"Length mismatch: candidates={len(candidates)} predictions={len(predictions)}"
        )

    grouped: dict[int, list[dict[str, object]]] = defaultdict(list)
    total_above = 0
    for row, prob in zip(candidates, predictions):
        if prob is None:
            continue
        entity_type = row.get("entity_type", "")
        if args.entity_type != "all" and entity_type != args.entity_type:
            continue
        record = {
            "user": int(row["user"]),
            "item": int(row["item"]),
            "entity_type": entity_type,
            "probability": float(prob),
            "_rank_score": float(prob) - args.pop_penalty * float(popularity.get(int(row["item"]), 0.0)),
        }
        if prob >= args.threshold:
            grouped[int(row["user"])].append(record)
            total_above += 1
        elif args.min_per_user > 0:
            record["_below_threshold"] = True
            grouped[int(row["user"])].append(record)

    selected: list[dict[str, object]] = []
    for user, rows in grouped.items():
        rows.sort(key=lambda r: (-float(r["_rank_score"]), -float(r["probability"]), int(r["item"])))
        above = [r for r in rows if not r.get("_below_threshold")]
        if args.topn_per_user > 0:
            keep = above[: args.topn_per_user]
        else:
            keep = above
        if args.min_per_user > 0 and len(keep) < args.min_per_user:
            seen = {(int(r["user"]), int(r["item"])) for r in keep}
            for row in rows:
                key = (int(row["user"]), int(row["item"]))
                if key in seen:
                    continue
                keep.append(row)
                seen.add(key)
                if len(keep) >= args.min_per_user:
                    break
        for row in keep:
            row.pop("_below_threshold", None)
        selected.extend(keep)

    if args.max_per_item > 0:
        item_counts: Counter[int] = Counter()
        capped = []
        for row in sorted(selected, key=lambda r: (-float(r["_rank_score"]), -float(r["probability"]), int(r["user"]), int(r["item"]))):
            item = int(row["item"])
            if item_counts[item] >= args.max_per_item:
                continue
            capped.append(row)
            item_counts[item] += 1
        selected = capped

    for row in selected:
        row.pop("_rank_score", None)
    selected.sort(key=lambda r: (int(r["user"]), r.get("entity_type", ""), -float(r["probability"]), int(r["item"])))
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["user", "item", "entity_type", "probability"])
        writer.writeheader()
        writer.writerows(selected)

    entity_counts = Counter(str(r["entity_type"]) for r in selected)
    user_counts = Counter(int(r["user"]) for r in selected)
    item_counts = Counter(int(r["item"]) for r in selected)
    probs = [float(r["probability"]) for r in selected]
    diagnostics = {
        "jsonl_path": str(args.jsonl_path),
        "top20_csv": str(args.top20_csv),
        "output_csv": str(args.output_csv),
        "threshold": args.threshold,
        "topn_per_user": args.topn_per_user,
        "min_per_user": args.min_per_user,
        "max_per_item": args.max_per_item,
        "pop_penalty": args.pop_penalty,
        "entity_type_filter": args.entity_type,
        "candidate_rows": len(candidates),
        "prediction_rows": len(predictions),
        "invalid_predictions": invalid,
        "rows_above_threshold_before_topn": total_above,
        "selected_rows": len(selected),
        "selected_users": len(user_counts),
        "selected_items": len(item_counts),
        "entity_type_counts": dict(entity_counts),
        "per_user_min": min(user_counts.values()) if user_counts else 0,
        "per_user_max": max(user_counts.values()) if user_counts else 0,
        "probability": {
            "min": min(probs) if probs else None,
            "max": max(probs) if probs else None,
            "mean": sum(probs) / len(probs) if probs else None,
        },
        "top_items": item_counts.most_common(20),
    }
    diag_path = args.diagnostics_json or args.output_csv.with_suffix(".diagnostics.json")
    diag_path.parent.mkdir(parents=True, exist_ok=True)
    diag_path.write_text(json.dumps(diagnostics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(diagnostics, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
