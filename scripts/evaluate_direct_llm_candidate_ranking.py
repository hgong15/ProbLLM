#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate direct LLM ranking on scored candidate pairs.")
    parser.add_argument("--pred_jsonl", type=Path, required=True)
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--output_json", type=Path, required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--candidate_pool", default="")
    parser.add_argument("--cold_object", choices=["item", "user"], default="item")
    parser.add_argument("--topk", type=int, default=20)
    return parser.parse_args()


def load_gt(data_dir: Path, filename: str) -> dict[int, list[int]]:
    df = pd.read_csv(data_dir / filename)
    gt: dict[int, list[int]] = defaultdict(list)
    for user, item in zip(df["user"].astype(int), df["item"].astype(int)):
        gt[int(user)].append(int(item))
    return gt


def dcg(labels: list[int]) -> float:
    return sum(rel / math.log2(idx + 2) for idx, rel in enumerate(labels))


def eval_split(rows: list[dict], gt: dict[int, list[int]], users: list[int] | None, topk: int) -> dict:
    if users is None:
        users = sorted(gt)
    by_user: dict[int, list[int]] = defaultdict(list)
    for row in sorted(rows, key=lambda r: (r["user"], -r["score"], r["idx"])):
        by_user[row["user"]].append(row["item"])

    precision = recall = ndcg = 0.0
    hits_total = gt_total = covered = n_users = 0
    for user in users:
        true = list(dict.fromkeys(gt.get(user, [])))
        if not true:
            continue
        n_users += 1
        gt_set = set(true)
        ranked = by_user.get(user, [])[:topk]
        if ranked:
            covered += 1
        hits = [1 if item in gt_set else 0 for item in ranked]
        hit_count = sum(hits)
        hits_total += hit_count
        gt_total += len(gt_set)
        precision += hit_count / float(topk)
        recall += hit_count / float(len(gt_set))
        ideal = dcg([1] * min(len(gt_set), topk))
        ndcg += dcg(hits) / ideal if ideal else 0.0

    return {
        "users": n_users,
        "users_with_candidates": covered,
        "candidate_user_coverage": covered / n_users if n_users else 0.0,
        "gt_pairs": gt_total,
        f"hits@{topk}": hits_total,
        f"precision@{topk}": precision / n_users if n_users else 0.0,
        f"recall@{topk}": recall / n_users if n_users else 0.0,
        f"ndcg@{topk}": ndcg / n_users if n_users else 0.0,
    }


def main() -> None:
    args = parse_args()
    rows = []
    with args.pred_jsonl.open(encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            if not line.strip():
                continue
            item = json.loads(line)
            rows.append(
                {
                    "idx": idx,
                    "user": int(item["user_id"]),
                    "item": int(item["item_id"]),
                    "entity_type": item.get("entity_type", ""),
                    "score": float(item["predict"]),
                }
            )

    strict_gt = load_gt(args.data_dir, f"cold_{args.cold_object}_test.csv")
    warmup_gt = load_gt(args.data_dir, "warmup_test.csv")
    overall_gt = load_gt(args.data_dir, "overall_test.csv")
    cold_union_gt: dict[int, list[int]] = defaultdict(list)
    for gt in (strict_gt, warmup_gt):
        for user, items in gt.items():
            cold_union_gt[user].extend(items)

    strict_rows = [row for row in rows if row["entity_type"] == "strict_cold"]
    warmup_rows = [row for row in rows if row["entity_type"] == "warmup"]
    metrics = {
        "strict_cold": eval_split(strict_rows, strict_gt, None, args.topk),
        "warmup": eval_split(warmup_rows, warmup_gt, None, args.topk),
        "strict_plus_warmup": eval_split(
            rows,
            cold_union_gt,
            sorted(set(strict_gt) | set(warmup_gt)),
            args.topk,
        ),
        "overall_scored_pairs_only": eval_split(rows, overall_gt, None, args.topk),
    }

    out = {
        "method": args.method,
        "candidate_pool": args.candidate_pool,
        "note": "Direct LLM ranking on scored strict-cold/warm-up candidate pairs only; no pseudo-interactions and no FinalUpdate.",
        "prediction_file": str(args.pred_jsonl),
        "num_scored_pairs": len(rows),
        "metrics": metrics,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Saved {args.output_json}")
    for split, metric in metrics.items():
        print(
            split,
            f'P/R/N={metric[f"precision@{args.topk}"]:.6f}/'
            f'{metric[f"recall@{args.topk}"]:.6f}/'
            f'{metric[f"ndcg@{args.topk}"]:.6f}',
            f'hits={metric[f"hits@{args.topk}"]}/{metric["gt_pairs"]}',
            f'coverage={metric["candidate_user_coverage"]:.4f}',
        )


if __name__ == "__main__":
    main()
