#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build CiteULike item-cold user candidates from arbitrary user/item embeddings."
    )
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--user_emb", type=Path, required=True)
    parser.add_argument("--item_emb", type=Path, required=True)
    parser.add_argument("--output_csv", type=Path, required=True)
    parser.add_argument("--meta_json", type=Path, default=None)
    parser.add_argument("--top_m", type=int, default=50)
    parser.add_argument("--item_batch_size", type=int, default=256)
    parser.add_argument("--method_name", default="embedding_item_topm")
    parser.add_argument("--normalize", action="store_true")
    return parser.parse_args()


def load_embedding(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        emb = np.asarray(np.load(path, mmap_mode="r"), dtype=np.float32)
    else:
        raise ValueError(f"Unsupported embedding format: {path}")
    if emb.ndim != 2:
        raise ValueError(f"Expected 2D embedding at {path}, got shape={emb.shape}")
    return emb


def maybe_normalize(emb: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return emb / norms


def load_target_items(data_dir: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    with (data_dir / "convert_dict.pkl").open("rb") as handle:
        para = pickle.load(handle)
    strict_items = np.asarray(
        para.get("strict_cold_item", para.get("cold_item", [])), dtype=np.int64
    )
    warmup_items = np.asarray(para.get("warmup_item", []), dtype=np.int64)
    return np.unique(strict_items), np.unique(warmup_items), para


def train_users_by_item(train_df: pd.DataFrame) -> dict[int, list[int]]:
    grouped: dict[int, list[int]] = defaultdict(list)
    for user, item in train_df[["user", "item"]].itertuples(index=False):
        grouped[int(item)].append(int(user))
    return grouped


def top_users_for_items(
    user_emb: np.ndarray,
    item_emb: np.ndarray,
    items: np.ndarray,
    entity_type: str,
    excluded_users: dict[int, list[int]],
    top_m: int,
    item_batch_size: int,
) -> list[dict[str, int | str | float]]:
    rows: list[dict[str, int | str | float]] = []
    users_take = min(top_m, user_emb.shape[0])
    for start in range(0, len(items), item_batch_size):
        batch_items = items[start : start + item_batch_size]
        scores = user_emb @ item_emb[batch_items].T
        for col, item in enumerate(batch_items.tolist()):
            item_scores = scores[:, col].astype(np.float32, copy=True)
            blocked = excluded_users.get(int(item))
            if blocked:
                blocked_idx = np.asarray(blocked, dtype=np.int64)
                blocked_idx = blocked_idx[blocked_idx < item_scores.shape[0]]
                item_scores[blocked_idx] = -np.inf
            if users_take < len(item_scores):
                top_idx = np.argpartition(-item_scores, users_take - 1)[:users_take]
            else:
                top_idx = np.arange(len(item_scores), dtype=np.int64)
            top_idx = top_idx[np.argsort(-item_scores[top_idx], kind="mergesort")]
            for rank, user in enumerate(top_idx[:users_take].tolist(), start=1):
                score = float(item_scores[user])
                if not np.isfinite(score):
                    continue
                rows.append(
                    {
                        "user": int(user),
                        "item": int(item),
                        "entity_type": entity_type,
                        "candidate_rank": int(rank),
                        "candidate_score": score,
                    }
                )
    return rows


def read_pairs(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["user", "item"])
    df = pd.read_csv(path)
    if df.empty:
        return pd.DataFrame(columns=["user", "item"])
    return df[["user", "item"]].astype({"user": int, "item": int})


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
    }
    for name, (entity_type, path) in files.items():
        gt = read_pairs(path)
        if gt.empty:
            stats[name] = {"pairs": 0, "hits": 0, "pair_recall": 0.0, "items": 0, "item_hit_rate": 0.0}
            continue
        hits = 0
        gt_items: set[int] = set()
        hit_items: set[int] = set()
        for user, item in gt[["user", "item"]].itertuples(index=False):
            item_id = int(item)
            gt_items.add(item_id)
            if (int(user), item_id, entity_type) in cand:
                hits += 1
                hit_items.add(item_id)
        stats[name] = {
            "pairs": int(len(gt)),
            "hits": int(hits),
            "pair_recall": float(hits / len(gt)) if len(gt) else 0.0,
            "items": int(len(gt_items)),
            "item_hit_rate": float(len(hit_items) / len(gt_items)) if gt_items else 0.0,
        }
    return stats


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir
    train_df = read_pairs(data_dir / "warm_emb.csv")
    if train_df.empty:
        raise FileNotFoundError(data_dir / "warm_emb.csv")

    user_emb = load_embedding(args.user_emb)
    item_emb = load_embedding(args.item_emb)
    if args.normalize:
        user_emb = maybe_normalize(user_emb)
        item_emb = maybe_normalize(item_emb)

    strict_items, warmup_items, para = load_target_items(data_dir)
    max_item = int(max(strict_items.max(initial=0), warmup_items.max(initial=0)))
    if max_item >= item_emb.shape[0]:
        raise ValueError(f"item_emb has {item_emb.shape[0]} rows but target item id {max_item} exists.")

    excluded = train_users_by_item(train_df)
    rows = []
    rows.extend(
        top_users_for_items(
            user_emb,
            item_emb,
            strict_items,
            "strict_cold",
            excluded,
            args.top_m,
            args.item_batch_size,
        )
    )
    rows.extend(
        top_users_for_items(
            user_emb,
            item_emb,
            warmup_items,
            "warmup",
            excluded,
            args.top_m,
            args.item_batch_size,
        )
    )

    out = pd.DataFrame(
        rows,
        columns=["user", "item", "entity_type", "candidate_rank", "candidate_score"],
    )
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output_csv, index=False)

    by_group = out.groupby(["item", "entity_type"]).size() if len(out) else pd.Series(dtype=int)
    meta = {
        "method": args.method_name,
        "data_dir": str(data_dir.resolve()),
        "top_m": args.top_m,
        "item_batch_size": args.item_batch_size,
        "normalize": bool(args.normalize),
        "user_emb": str(args.user_emb.resolve()),
        "item_emb": str(args.item_emb.resolve()),
        "user_emb_shape": list(user_emb.shape),
        "item_emb_shape": list(item_emb.shape),
        "train_interactions": int(len(train_df)),
        "strict_cold_items": int(len(strict_items)),
        "warmup_items": int(len(warmup_items)),
        "warm_items": int(len(para.get("warm_item", []))),
        "rows": int(len(out)),
        "items": int(out["item"].nunique()) if len(out) else 0,
        "users": int(out["user"].nunique()) if len(out) else 0,
        "groups_with_exact_top_m": int((by_group == args.top_m).sum()) if len(by_group) else 0,
        "groups_total": int(len(by_group)),
        "entity_type_counts": out["entity_type"].value_counts().to_dict() if len(out) else {},
        "candidate_hit": candidate_hit_stats(out, data_dir),
        "output_csv": str(args.output_csv.resolve()),
        "candidate_policy": (
            "For each strict-cold/warm-up item, rank users by user_emb dot item_emb; "
            "remove users already connected to that item in warm_emb.csv."
        ),
    }
    meta_path = args.meta_json or args.output_csv.with_suffix(".meta.json")
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2), flush=True)


if __name__ == "__main__":
    main()
