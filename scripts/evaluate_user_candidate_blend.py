#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate per-user candidate ranking with embedding/LLM blend scores.")
    parser.add_argument("--top_csv", type=Path, required=True)
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--output_json", type=Path, required=True)
    parser.add_argument("--prediction_jsonl", type=Path, default=None)
    parser.add_argument("--llm_weights", default="0.00,0.05,0.10,0.25,0.50,0.75,1.00")
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--method", default="candidate_blend")
    return parser.parse_args()


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


def read_predictions(path: Path | None, expected: int) -> tuple[list[float | None], int]:
    if path is None:
        return [0.5] * expected, 0
    predictions = []
    invalid = 0
    with path.open("r", encoding="utf-8") as handle:
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
    if len(predictions) != expected:
        raise ValueError(f"Length mismatch: candidates={expected} predictions={len(predictions)}")
    return predictions, invalid


def read_pairs(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.empty:
        return pd.DataFrame(columns=["user", "item"])
    return df[["user", "item"]].astype({"user": int, "item": int})


def load_gt(data_dir: Path, filename: str) -> dict[int, list[int]]:
    df = read_pairs(data_dir / filename)
    gt: dict[int, list[int]] = defaultdict(list)
    for user, item in df[["user", "item"]].itertuples(index=False):
        gt[int(user)].append(int(item))
    return gt


def normalize(values: list[float]) -> list[float]:
    if not values:
        return []
    lo = min(values)
    hi = max(values)
    if hi <= lo:
        return [0.5 for _ in values]
    return [(value - lo) / (hi - lo) for value in values]


def dcg(labels: list[int]) -> float:
    return sum(rel / math.log2(idx + 2) for idx, rel in enumerate(labels))


def eval_split(groups, gt: dict[int, list[int]], topk: int) -> dict:
    precision = recall = ndcg_value = 0.0
    hits_total = gt_total = covered = n_users = 0
    for user in sorted(gt):
        true = list(dict.fromkeys(gt.get(user, [])))
        if not true:
            continue
        n_users += 1
        gt_set = set(true)
        ranked = groups.get(user, [])[:topk]
        if ranked:
            covered += 1
        hits = [1 if item in gt_set else 0 for item in ranked]
        hit_count = sum(hits)
        hits_total += hit_count
        gt_total += len(gt_set)
        precision += hit_count / float(topk)
        recall += hit_count / float(len(gt_set))
        ideal = dcg([1] * min(len(gt_set), topk))
        ndcg_value += dcg(hits) / ideal if ideal else 0.0
    return {
        "users": int(n_users),
        "users_with_candidates": int(covered),
        "candidate_user_coverage": covered / n_users if n_users else 0.0,
        "gt_pairs": int(gt_total),
        f"hits@{topk}": int(hits_total),
        f"precision@{topk}": precision / n_users if n_users else 0.0,
        f"recall@{topk}": recall / n_users if n_users else 0.0,
        f"ndcg@{topk}": ndcg_value / n_users if n_users else 0.0,
    }


def parse_weights(text: str) -> list[float]:
    return [float(part) for part in text.split(",") if part.strip()]


def build_ranked_groups(rows: list[dict], llm_weight: float) -> dict[tuple[str, int], list[int]]:
    grouped_rows: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in rows:
        grouped_rows[(row["entity_type"], row["user"])].append(row)

    ranked: dict[tuple[str, int], list[int]] = {}
    emb_weight = 1.0 - llm_weight
    for key, group in grouped_rows.items():
        llm_norm = normalize([row["llm_probability"] for row in group])
        emb_norm = normalize([row["embedding_score"] for row in group])
        scored = []
        for row, llm_value, emb_value in zip(group, llm_norm, emb_norm):
            score = llm_weight * llm_value + emb_weight * emb_value
            scored.append((score, llm_value, emb_value, -row["candidate_rank"], -row["item"], row["item"]))
        scored.sort(reverse=True)
        ranked[key] = [int(entry[-1]) for entry in scored]
    return ranked


def main() -> None:
    args = parse_args()
    weights = parse_weights(args.llm_weights)
    candidates = pd.read_csv(args.top_csv)
    predictions, invalid = read_predictions(args.prediction_jsonl, len(candidates))

    rows = []
    for row, prob in zip(candidates.to_dict("records"), predictions):
        if prob is None:
            continue
        rows.append(
            {
                "user": int(row["user"]),
                "item": int(row["item"]),
                "entity_type": str(row.get("entity_type", "")),
                "candidate_rank": int(row.get("candidate_rank", 0)),
                "embedding_score": float(row.get("candidate_score", -float(row.get("candidate_rank", 0)))),
                "llm_probability": float(prob),
            }
        )

    gt = {
        "strict_cold": load_gt(args.data_dir, "cold_item_test.csv"),
        "warmup": load_gt(args.data_dir, "warmup_test.csv"),
        "warm": load_gt(args.data_dir, "warm_test.csv"),
    }
    cold_union_gt: dict[int, list[int]] = defaultdict(list)
    for split in ("strict_cold", "warmup"):
        for user, items in gt[split].items():
            cold_union_gt[user].extend(items)

    results = []
    for weight in weights:
        ranked = build_ranked_groups(rows, weight)
        strict_groups = {user: items for (entity_type, user), items in ranked.items() if entity_type == "strict_cold"}
        warmup_groups = {user: items for (entity_type, user), items in ranked.items() if entity_type == "warmup"}
        warm_groups = {user: items for (entity_type, user), items in ranked.items() if entity_type == "warm"}
        union_groups: dict[int, list[int]] = defaultdict(list)
        for user, items in strict_groups.items():
            union_groups[user].extend(items)
        for user, items in warmup_groups.items():
            union_groups[user].extend(items)
        results.append(
            {
                "llm_weight": float(weight),
                "embedding_weight": float(1.0 - weight),
                "metrics": {
                    "strict_cold": eval_split(strict_groups, gt["strict_cold"], args.topk),
                    "warmup": eval_split(warmup_groups, gt["warmup"], args.topk),
                    "warm": eval_split(warm_groups, gt["warm"], args.topk),
                    "strict_plus_warmup": eval_split(union_groups, cold_union_gt, args.topk),
                },
            }
        )

    out = {
        "method": args.method,
        "top_csv": str(args.top_csv.resolve()),
        "prediction_jsonl": str(args.prediction_jsonl.resolve()) if args.prediction_jsonl else None,
        "candidate_rows": int(len(candidates)),
        "valid_rows": int(len(rows)),
        "invalid_predictions": int(invalid),
        "topk": int(args.topk),
        "results": results,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Saved {args.output_json}")
    for row in results:
        metrics = row["metrics"]["strict_plus_warmup"]
        print(
            f"llm_weight={row['llm_weight']:.2f}",
            f"P/R/N={metrics[f'precision@{args.topk}']:.4f}/"
            f"{metrics[f'recall@{args.topk}']:.4f}/"
            f"{metrics[f'ndcg@{args.topk}']:.4f}",
        )


if __name__ == "__main__":
    main()
