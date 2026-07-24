#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build user-side text-neighbor top-M item candidates for cold-user LLM augmentation."
    )
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--embedding_path", type=Path, default=None)
    parser.add_argument("--output_csv", type=Path, required=True)
    parser.add_argument("--meta_json", type=Path, default=None)
    parser.add_argument("--top_m", type=int, default=20)
    parser.add_argument("--neighbor_k", type=int, default=100)
    parser.add_argument("--score_agg", choices=["sum", "max"], default="sum")
    return parser.parse_args()


def read_pairs(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["user", "item"])
    df = pd.read_csv(path)
    if df.empty:
        return pd.DataFrame(columns=["user", "item"])
    return df[["user", "item"]].astype({"user": int, "item": int})


def target_user_types(data_dir: Path) -> dict[int, str]:
    target: dict[int, str] = {}
    for name in ("cold_user_val.csv", "cold_user_test.csv"):
        for user in read_pairs(data_dir / name)["user"].unique().tolist():
            target[int(user)] = "strict_cold"
    for name in ("warmup_val.csv", "warmup_test.csv"):
        for user in read_pairs(data_dir / name)["user"].unique().tolist():
            target[int(user)] = "warmup"
    return target


def build_train_indexes(train_df: pd.DataFrame):
    items_by_user: dict[int, list[int]] = defaultdict(list)
    item_sets_by_user: dict[int, set[int]] = defaultdict(set)
    for user, item in train_df[["user", "item"]].itertuples(index=False):
        user_id = int(user)
        item_id = int(item)
        items_by_user[user_id].append(item_id)
        item_sets_by_user[user_id].add(item_id)
    return items_by_user, item_sets_by_user


def choose_items_for_user(
    user: int,
    user_emb: torch.Tensor,
    warm_users: torch.Tensor,
    warm_emb: torch.Tensor,
    items_by_user: dict[int, list[int]],
    item_sets_by_user: dict[int, set[int]],
    neighbor_k: int,
    top_m: int,
    score_agg: str,
) -> tuple[list[tuple[int, float]], list[int]]:
    sims = (user_emb[user : user + 1] @ warm_emb.t()).squeeze(0)
    same_user = (warm_users == user).nonzero(as_tuple=False).flatten()
    if same_user.numel():
        sims[same_user] = -torch.inf

    take = min(neighbor_k, int(warm_users.numel()))
    values, local_indices = torch.topk(sims, k=take)
    blocked = item_sets_by_user.get(user, set())
    scores: dict[int, float] = defaultdict(float)
    neighbors: list[int] = []
    for sim, local in zip(values.tolist(), local_indices.tolist()):
        if not np.isfinite(sim):
            continue
        warm_user = int(warm_users[local])
        neighbors.append(warm_user)
        contribution = max(float(sim), 0.0)
        for item in items_by_user.get(warm_user, []):
            if item in blocked:
                continue
            if score_agg == "max":
                scores[item] = max(scores[item], contribution)
            else:
                scores[item] += contribution

    ranked = sorted(scores.items(), key=lambda row: (-row[1], row[0]))[:top_m]
    return ranked, neighbors


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir
    emb_path = args.embedding_path or data_dir / "llm_user_content_emb.pt"
    user_emb = torch.load(emb_path, map_location="cpu").float()
    user_emb = F.normalize(user_emb, p=2, dim=1)

    train_df = read_pairs(data_dir / "warm_emb.csv")
    items_by_user, item_sets_by_user = build_train_indexes(train_df)
    warm_users = torch.as_tensor(sorted(items_by_user.keys()), dtype=torch.long)
    if warm_users.numel() == 0:
        raise ValueError("No warm training users found in warm_emb.csv")
    warm_emb = user_emb[warm_users]
    targets = target_user_types(data_dir)
    if not targets:
        raise ValueError("No strict-cold/warm-up target users found.")

    rows = []
    short_users = []
    neighbor_examples = {}
    for user in tqdm(sorted(targets), desc="Selecting user text-neighbor candidates"):
        ranked, neighbors = choose_items_for_user(
            int(user),
            user_emb,
            warm_users,
            warm_emb,
            items_by_user,
            item_sets_by_user,
            args.neighbor_k,
            args.top_m,
            args.score_agg,
        )
        if len(neighbor_examples) < 5:
            neighbor_examples[str(user)] = neighbors[:10]
        if len(ranked) < args.top_m:
            short_users.append({"user": int(user), "entity_type": targets[int(user)], "candidates": len(ranked)})
        for item, score in ranked:
            rows.append(
                {
                    "user": int(user),
                    "item": int(item),
                    "entity_type": targets[int(user)],
                    "candidate_score": float(score),
                }
            )

    df = pd.DataFrame(rows, columns=["user", "item", "entity_type", "candidate_score"])
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_csv, index=False)
    meta = {
        "method": "User text-neighbor top-M",
        "definition": (
            "For each strict-cold/warm-up user, find nearest warm training users by frozen "
            "LLaMA user-profile cosine similarity, aggregate those warm users' training items, "
            "and select top-M items. No LoRA or pairwise LLM is used for candidate generation."
        ),
        "dataset": data_dir.name,
        "seed": args.seed,
        "top_m": args.top_m,
        "neighbor_k": args.neighbor_k,
        "score_agg": args.score_agg,
        "train_interactions": int(len(train_df)),
        "warm_source_users": int(warm_users.numel()),
        "target_users": int(len(targets)),
        "strict_rows": int((df["entity_type"] == "strict_cold").sum()) if not df.empty else 0,
        "warmup_rows": int((df["entity_type"] == "warmup").sum()) if not df.empty else 0,
        "short_users": short_users[:20],
        "short_user_count": len(short_users),
        "neighbor_examples": neighbor_examples,
        "user_embedding": str(emb_path),
        "rows": int(len(df)),
        "users": int(df["user"].nunique()) if not df.empty else 0,
        "items": int(df["item"].nunique()) if not df.empty else 0,
        "output_csv": str(args.output_csv),
    }
    meta_path = args.meta_json or args.output_csv.with_suffix(".meta.json")
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2), flush=True)


if __name__ == "__main__":
    main()
