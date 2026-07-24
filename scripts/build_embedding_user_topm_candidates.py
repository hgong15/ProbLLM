#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build item-cold candidates by ranking target items for each user."
    )
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--user_emb", type=Path, required=True)
    parser.add_argument("--item_emb", type=Path, required=True)
    parser.add_argument("--output_csv", type=Path, required=True)
    parser.add_argument("--meta_json", type=Path, default=None)
    parser.add_argument("--top_m", type=int, default=50)
    parser.add_argument("--user_batch_size", type=int, default=256)
    parser.add_argument("--method_name", default="embedding_user_topm")
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument(
        "--target_splits",
        default="strict_cold,warmup",
        help="Comma-separated candidate item groups to rank: strict_cold,warmup,warm.",
    )
    parser.add_argument(
        "--users",
        choices=["all", "eval"],
        default="all",
        help="all ranks candidates for every user; eval only ranks users in val/test files.",
    )
    return parser.parse_args()


def load_embedding(path: Path) -> np.ndarray:
    if path.suffix != ".npy":
        raise ValueError(f"Unsupported embedding format: {path}")
    emb = np.asarray(np.load(path, mmap_mode="r"), dtype=np.float32)
    if emb.ndim != 2:
        raise ValueError(f"Expected 2D embedding at {path}, got shape={emb.shape}")
    return emb


def maybe_normalize(emb: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    return emb / np.maximum(norms, 1e-12)


def read_pairs(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["user", "item"])
    df = pd.read_csv(path)
    if df.empty:
        return pd.DataFrame(columns=["user", "item"])
    return df[["user", "item"]].astype({"user": int, "item": int})


def parse_target_splits(text: str) -> list[str]:
    splits = [part.strip() for part in text.split(",") if part.strip()]
    allowed = {"strict_cold", "warmup", "warm"}
    unknown = sorted(set(splits) - allowed)
    if unknown:
        raise ValueError(f"Unsupported target_splits: {unknown}; allowed={sorted(allowed)}")
    return splits


def load_target_items(data_dir: Path) -> tuple[dict[str, np.ndarray], dict]:
    with (data_dir / "convert_dict.pkl").open("rb") as handle:
        para = pickle.load(handle)
    strict_items = np.asarray(
        para.get("strict_cold_item", para.get("cold_item", [])), dtype=np.int64
    )
    warmup_items = np.asarray(para.get("warmup_item", []), dtype=np.int64)
    warm_items = np.asarray(para.get("warm_item", []), dtype=np.int64)
    return {
        "strict_cold": np.unique(strict_items),
        "warmup": np.unique(warmup_items),
        "warm": np.unique(warm_items),
    }, para


def excluded_items_by_user(data_dir: Path) -> dict[int, set[int]]:
    excluded: dict[int, set[int]] = defaultdict(set)
    for name in ("warm_emb.csv", "warmup_support.csv"):
        df = read_pairs(data_dir / name)
        for user, item in df[["user", "item"]].itertuples(index=False):
            excluded[int(user)].add(int(item))
    return excluded


def eval_users(data_dir: Path) -> np.ndarray:
    users: set[int] = set()
    for name in (
        "cold_item_val.csv",
        "cold_item_test.csv",
        "warmup_val.csv",
        "warmup_test.csv",
        "warm_val.csv",
        "warm_test.csv",
    ):
        df = read_pairs(data_dir / name)
        users.update(map(int, df["user"].tolist()))
    return np.asarray(sorted(users), dtype=np.int64)


def top_items_for_users(
    user_emb: np.ndarray,
    item_emb: np.ndarray,
    users: np.ndarray,
    items: np.ndarray,
    entity_type: str,
    excluded: dict[int, set[int]],
    top_m: int,
    user_batch_size: int,
) -> list[dict[str, int | str | float]]:
    rows: list[dict[str, int | str | float]] = []
    if len(items) == 0 or len(users) == 0:
        return rows
    items_take = min(top_m, len(items))
    target_item_emb = item_emb[items]
    item_to_col = {int(item): idx for idx, item in enumerate(items.tolist())}
    for start in range(0, len(users), user_batch_size):
        batch_users = users[start : start + user_batch_size]
        scores = user_emb[batch_users] @ target_item_emb.T
        for row_idx, user in enumerate(batch_users.tolist()):
            user_scores = scores[row_idx].astype(np.float32, copy=True)
            blocked = excluded.get(int(user), set())
            if blocked:
                blocked_cols = [item_to_col[item] for item in blocked if item in item_to_col]
                if blocked_cols:
                    user_scores[np.asarray(blocked_cols, dtype=np.int64)] = -np.inf
            if items_take < len(user_scores):
                top_idx = np.argpartition(-user_scores, items_take - 1)[:items_take]
            else:
                top_idx = np.arange(len(user_scores), dtype=np.int64)
            top_idx = top_idx[np.argsort(-user_scores[top_idx], kind="mergesort")]
            for rank, col in enumerate(top_idx[:items_take].tolist(), start=1):
                score = float(user_scores[col])
                if not np.isfinite(score):
                    continue
                rows.append(
                    {
                        "user": int(user),
                        "item": int(items[col]),
                        "entity_type": entity_type,
                        "candidate_rank": int(rank),
                        "candidate_score": score,
                    }
                )
    return rows


def dcg(labels: list[int]) -> float:
    return sum(rel / math.log2(idx + 2) for idx, rel in enumerate(labels))


def eval_candidate_ranking(rows: pd.DataFrame, data_dir: Path, topk: int) -> dict[str, dict]:
    split_specs = {
        "strict_cold": ("strict_cold", data_dir / "cold_item_test.csv"),
        "warmup": ("warmup", data_dir / "warmup_test.csv"),
        "warm": ("warm", data_dir / "warm_test.csv"),
    }
    metrics: dict[str, dict] = {}
    for split, (entity_type, gt_path) in split_specs.items():
        gt_df = read_pairs(gt_path)
        gt_by_user = {
            int(user): list(map(int, group["item"].tolist()))
            for user, group in gt_df.groupby("user")
        }
        split_rows = rows[rows["entity_type"] == entity_type]
        ranked_by_user: dict[int, list[int]] = defaultdict(list)
        for row in split_rows.sort_values(
            ["user", "candidate_rank"], ascending=[True, True], kind="mergesort"
        ).itertuples(index=False):
            ranked_by_user[int(row.user)].append(int(row.item))

        precision = recall = ndcg_value = 0.0
        hits_total = gt_total = covered = n_users = 0
        for user, true_items in gt_by_user.items():
            true_unique = list(dict.fromkeys(true_items))
            if not true_unique:
                continue
            n_users += 1
            gt_set = set(true_unique)
            ranked = ranked_by_user.get(user, [])[:topk]
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
        metrics[split] = {
            "users": int(n_users),
            "users_with_candidates": int(covered),
            "candidate_user_coverage": covered / n_users if n_users else 0.0,
            "gt_pairs": int(gt_total),
            f"hits@{topk}": int(hits_total),
            f"precision@{topk}": precision / n_users if n_users else 0.0,
            f"recall@{topk}": recall / n_users if n_users else 0.0,
            f"ndcg@{topk}": ndcg_value / n_users if n_users else 0.0,
        }
    return metrics


def candidate_hit_stats(topm: pd.DataFrame, data_dir: Path) -> dict[str, dict[str, float | int]]:
    stats: dict[str, dict[str, float | int]] = {}
    cand = {
        (int(row.user), int(row.item), str(row.entity_type))
        for row in topm[["user", "item", "entity_type"]].itertuples(index=False)
    }
    files = {
        "strict_cold_val": ("strict_cold", data_dir / "cold_item_val.csv"),
        "strict_cold_test": ("strict_cold", data_dir / "cold_item_test.csv"),
        "warmup_val": ("warmup", data_dir / "warmup_val.csv"),
        "warmup_test": ("warmup", data_dir / "warmup_test.csv"),
        "warm_val": ("warm", data_dir / "warm_val.csv"),
        "warm_test": ("warm", data_dir / "warm_test.csv"),
    }
    for name, (entity_type, path) in files.items():
        gt = read_pairs(path)
        if gt.empty:
            stats[name] = {"pairs": 0, "hits": 0, "pair_recall": 0.0, "users": 0, "user_hit_rate": 0.0}
            continue
        hits = 0
        gt_users: set[int] = set()
        hit_users: set[int] = set()
        for user, item in gt[["user", "item"]].itertuples(index=False):
            user_id = int(user)
            gt_users.add(user_id)
            if (user_id, int(item), entity_type) in cand:
                hits += 1
                hit_users.add(user_id)
        stats[name] = {
            "pairs": int(len(gt)),
            "hits": int(hits),
            "pair_recall": float(hits / len(gt)) if len(gt) else 0.0,
            "users": int(len(gt_users)),
            "user_hit_rate": float(len(hit_users) / len(gt_users)) if gt_users else 0.0,
        }
    return stats


def main() -> None:
    args = parse_args()
    user_emb = load_embedding(args.user_emb)
    item_emb = load_embedding(args.item_emb)
    if args.normalize:
        user_emb = maybe_normalize(user_emb)
        item_emb = maybe_normalize(item_emb)

    target_splits = parse_target_splits(args.target_splits)
    item_groups, para = load_target_items(args.data_dir)
    selected_items = [item_groups[split] for split in target_splits if len(item_groups[split])]
    max_item = int(max((items.max(initial=0) for items in selected_items), default=0))
    if max_item >= item_emb.shape[0]:
        raise ValueError(f"item_emb has {item_emb.shape[0]} rows but target item id {max_item} exists.")

    if args.users == "all":
        users = np.arange(user_emb.shape[0], dtype=np.int64)
    else:
        users = eval_users(args.data_dir)
    excluded = excluded_items_by_user(args.data_dir)

    rows = []
    for split in target_splits:
        rows.extend(
            top_items_for_users(
                user_emb,
                item_emb,
                users,
                item_groups[split],
                split,
                excluded,
                args.top_m,
                args.user_batch_size,
            )
        )

    out = pd.DataFrame(
        rows,
        columns=["user", "item", "entity_type", "candidate_rank", "candidate_score"],
    )
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output_csv, index=False)

    meta = {
        "method": args.method_name,
        "data_dir": str(args.data_dir.resolve()),
        "top_m": int(args.top_m),
        "target_splits": target_splits,
        "user_batch_size": int(args.user_batch_size),
        "normalize": bool(args.normalize),
        "users_policy": args.users,
        "user_emb": str(args.user_emb.resolve()),
        "item_emb": str(args.item_emb.resolve()),
        "user_emb_shape": list(user_emb.shape),
        "item_emb_shape": list(item_emb.shape),
        "strict_cold_items": int(len(item_groups["strict_cold"])),
        "warmup_items": int(len(item_groups["warmup"])),
        "warm_items": int(len(item_groups["warm"])),
        "candidate_users": int(len(users)),
        "rows": int(len(out)),
        "items": int(out["item"].nunique()) if len(out) else 0,
        "users": int(out["user"].nunique()) if len(out) else 0,
        "entity_type_counts": out["entity_type"].value_counts().to_dict() if len(out) else {},
        "candidate_hit": candidate_hit_stats(out, args.data_dir),
        "direct_candidate_metrics": eval_candidate_ranking(out, args.data_dir, topk=20),
        "candidate_policy": (
            f"For each user, rank {', '.join(target_splits)} items separately by "
            "user_emb dot item_emb; remove known warm_emb/warmup_support interactions."
        ),
        "output_csv": str(args.output_csv.resolve()),
    }
    meta_path = args.meta_json or args.output_csv.with_suffix(".meta.json")
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2), flush=True)


if __name__ == "__main__":
    main()
