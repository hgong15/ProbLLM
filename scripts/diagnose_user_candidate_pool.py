#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit user-side candidate pools against held-out strict/warm-up positives."
    )
    parser.add_argument("--candidate_csv", type=Path, required=True)
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--output_json", type=Path, required=True)
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--embedding_user_path", type=Path)
    parser.add_argument("--embedding_item_path", type=Path)
    return parser.parse_args()


def read_user_items(path: Path) -> dict[int, list[int]]:
    df = pd.read_csv(path)
    if not {"user", "item"} <= set(df.columns):
        raise ValueError(f"{path} must contain user,item columns")
    out: dict[int, list[int]] = defaultdict(list)
    for user, item in df[["user", "item"]].itertuples(index=False):
        out[int(user)].append(int(item))
    return dict(out)


def load_candidates(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"user", "item", "entity_type"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing required columns: {sorted(missing)}")
    if "probability" not in df.columns:
        df["probability"] = np.nan
    df["_row_order"] = np.arange(len(df), dtype=np.int64)
    return df


def dcg_from_hits(hits: Iterable[float], topk: int) -> float:
    arr = np.asarray(list(hits), dtype=np.float64)[:topk]
    if len(arr) == 0:
        return 0.0
    discounts = 1.0 / np.log2(np.arange(2, len(arr) + 2))
    return float(np.sum(arr * discounts))


def idcg(num_positive: int, topk: int) -> float:
    n = min(int(num_positive), int(topk))
    if n <= 0:
        return 1.0
    return float(np.sum(1.0 / np.log2(np.arange(2, n + 2))))


def evaluate_group(candidates: pd.DataFrame, gt: dict[int, list[int]], entity_type: str, topk: int) -> dict:
    subset = candidates[candidates["entity_type"].astype(str) == entity_type].copy()
    if subset.empty:
        return {
            "entity_type": entity_type,
            "users": len(gt),
            "candidate_users": 0,
            "candidate_rows": 0,
            "ranked": {"recall@20": 0.0, "ndcg@20": 0.0, "precision@20": 0.0},
            "oracle": {"recall@20": 0.0, "ndcg@20": 0.0, "precision@20": 0.0},
            "hit": {"pairs": sum(len(v) for v in gt.values()), "hits": 0, "pair_recall": 0.0, "user_hit_rate": 0.0},
        }

    ordered = subset.sort_values(["user", "probability", "_row_order"], ascending=[True, False, True])
    per_user_candidates = {
        int(user): [int(item) for item in group["item"].tolist()]
        for user, group in ordered.groupby("user", sort=False)
    }

    ranked_recall = 0.0
    ranked_ndcg = 0.0
    ranked_precision = 0.0
    oracle_recall = 0.0
    oracle_ndcg = 0.0
    oracle_precision = 0.0
    hit_pairs = 0
    hit_users = 0
    gt_pairs = 0

    for user, positives in gt.items():
        pos_set = set(int(x) for x in positives)
        gt_pairs += len(pos_set)
        cand = per_user_candidates.get(int(user), [])
        cand_unique = []
        seen = set()
        for item in cand:
            if item not in seen:
                seen.add(item)
                cand_unique.append(item)

        hits_all = [item for item in cand_unique if item in pos_set]
        hit_pairs += len(hits_all)
        if hits_all:
            hit_users += 1

        top = cand_unique[:topk]
        hits_top = [1.0 if item in pos_set else 0.0 for item in top]
        ranked_hits = float(sum(hits_top))
        ranked_recall += ranked_hits / max(len(pos_set), 1)
        ranked_precision += ranked_hits / float(topk)
        ranked_ndcg += dcg_from_hits(hits_top, topk) / idcg(len(pos_set), topk)

        oracle_hits = min(topk, len(hits_all))
        oracle_recall += oracle_hits / max(len(pos_set), 1)
        oracle_precision += oracle_hits / float(topk)
        oracle_ndcg += idcg(oracle_hits, topk) / idcg(len(pos_set), topk)

    n_users = max(len(gt), 1)
    return {
        "entity_type": entity_type,
        "users": int(len(gt)),
        "candidate_users": int(len(per_user_candidates)),
        "candidate_rows": int(len(subset)),
        "per_user_candidate_count": subset.groupby("user").size().describe().to_dict(),
        "ranked": {
            f"recall@{topk}": ranked_recall / n_users,
            f"ndcg@{topk}": ranked_ndcg / n_users,
            f"precision@{topk}": ranked_precision / n_users,
        },
        "oracle": {
            f"recall@{topk}": oracle_recall / n_users,
            f"ndcg@{topk}": oracle_ndcg / n_users,
            f"precision@{topk}": oracle_precision / n_users,
        },
        "hit": {
            "pairs": int(gt_pairs),
            "hits": int(hit_pairs),
            "pair_recall": float(hit_pairs / gt_pairs) if gt_pairs else 0.0,
            "user_hit_rate": float(hit_users / len(gt)) if gt else 0.0,
        },
    }


def main() -> None:
    args = parse_args()
    candidates = load_candidates(args.candidate_csv)
    strict_gt = read_user_items(args.data_dir / "cold_user_test.csv")
    warmup_gt = read_user_items(args.data_dir / "warmup_test.csv")

    summary = {
        "candidate_csv": str(args.candidate_csv.resolve()),
        "data_dir": str(args.data_dir.resolve()),
        "embedding_user_path": str(args.embedding_user_path.resolve()) if args.embedding_user_path else None,
        "embedding_item_path": str(args.embedding_item_path.resolve()) if args.embedding_item_path else None,
        "topk": int(args.topk),
        "rows": int(len(candidates)),
        "unique_users": int(candidates["user"].nunique()),
        "unique_items": int(candidates["item"].nunique()),
        "entity_type_counts": candidates["entity_type"].astype(str).value_counts().to_dict(),
        "strict_cold": evaluate_group(candidates, strict_gt, "strict_cold", args.topk),
        "warmup": evaluate_group(candidates, warmup_gt, "warmup", args.topk),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
