#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build Book-Crossing cold-user top-M item candidates from pure user-content "
            "nearest warm users."
        )
    )
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--embedding_path", type=Path, default=None)
    parser.add_argument("--output_csv", type=Path, required=True)
    parser.add_argument("--meta_json", type=Path, default=None)
    parser.add_argument("--top_m", type=int, default=20)
    parser.add_argument("--neighbor_k", type=int, default=100)
    parser.add_argument("--score_agg", choices=["sum", "max"], default="sum")
    parser.add_argument("--sim_batch_size", type=int, default=512)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--limit_users", type=int, default=0)
    return parser.parse_args()


def read_pairs(path: Path, empty_ok: bool = True) -> pd.DataFrame:
    if not path.exists():
        if empty_ok:
            return pd.DataFrame(columns=["user", "item"])
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    if df.empty:
        return pd.DataFrame(columns=["user", "item"])
    return df[["user", "item"]].astype({"user": int, "item": int})


def target_user_types(data_dir: Path) -> dict[int, str]:
    targets: dict[int, str] = {}
    for name in ("cold_user_val.csv", "cold_user_test.csv"):
        for user in read_pairs(data_dir / name)["user"].unique().tolist():
            targets[int(user)] = "strict_cold"
    for name in ("warmup_val.csv", "warmup_test.csv"):
        for user in read_pairs(data_dir / name)["user"].unique().tolist():
            targets[int(user)] = "warmup"
    return targets


def build_train_indexes(train_df: pd.DataFrame):
    items_by_user: dict[int, list[int]] = defaultdict(list)
    item_counter: Counter[int] = Counter()
    for user, item in train_df[["user", "item"]].itertuples(index=False):
        user_id = int(user)
        item_id = int(item)
        items_by_user[user_id].append(item_id)
        item_counter[item_id] += 1
    popular_items = [item for item, _ in item_counter.most_common()]
    return items_by_user, popular_items


def observed_by_user(data_dir: Path) -> dict[int, set[int]]:
    observed: dict[int, set[int]] = defaultdict(set)
    for name in ("warm_emb.csv", "warm_train.csv", "warmup_support.csv"):
        path = data_dir / name
        if not path.exists():
            continue
        df = read_pairs(path)
        for user, item in df[["user", "item"]].itertuples(index=False):
            observed[int(user)].add(int(item))
    return observed


def load_embedding(path: Path) -> torch.Tensor:
    if path.suffix == ".pt":
        emb = torch.load(path, map_location="cpu").float()
    elif path.suffix == ".npy":
        emb = torch.from_numpy(np.asarray(np.load(path, mmap_mode="r"), dtype=np.float32))
    else:
        raise ValueError(f"Unsupported embedding format: {path}")
    if emb.ndim != 2:
        raise ValueError(f"Expected 2D embedding at {path}, got shape={tuple(emb.shape)}")
    return F.normalize(emb.float(), p=2, dim=1)


def candidate_hit_stats(top20: pd.DataFrame, data_dir: Path) -> dict[str, dict[str, float | int]]:
    stats: dict[str, dict[str, float | int]] = {}
    by_user = top20.groupby("user")["item"].agg(set).to_dict() if len(top20) else {}
    files = {
        "strict_cold_val": ("strict_cold", data_dir / "cold_user_val.csv"),
        "strict_cold_test": ("strict_cold", data_dir / "cold_user_test.csv"),
        "warmup_val": ("warmup", data_dir / "warmup_val.csv"),
        "warmup_test": ("warmup", data_dir / "warmup_test.csv"),
    }
    for name, (entity_type, path) in files.items():
        gt = read_pairs(path)
        if gt.empty:
            stats[name] = {"users": 0, "gt_pairs": 0, "hits": 0, "recall": 0.0, "user_hit_rate": 0.0}
            continue
        cand_users = set(top20.loc[top20["entity_type"] == entity_type, "user"].astype(int).tolist())
        hits = 0
        user_hit: set[int] = set()
        gt_users: set[int] = set()
        for user, item in gt[["user", "item"]].itertuples(index=False):
            user_id = int(user)
            item_id = int(item)
            gt_users.add(user_id)
            if user_id in cand_users and item_id in by_user.get(user_id, set()):
                hits += 1
                user_hit.add(user_id)
        stats[name] = {
            "users": len(gt_users),
            "gt_pairs": int(len(gt)),
            "hits": int(hits),
            "recall": float(hits / len(gt)) if len(gt) else 0.0,
            "user_hit_rate": float(len(user_hit) / len(gt_users)) if gt_users else 0.0,
        }
    return stats


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir
    emb_path = args.embedding_path or data_dir / "book-crossing_user_content.npy"
    user_emb = load_embedding(emb_path)

    train_df = read_pairs(data_dir / "warm_emb.csv", empty_ok=False)
    items_by_user, popular_items = build_train_indexes(train_df)
    blocked_by_user = observed_by_user(data_dir)
    warm_users = torch.as_tensor(sorted(items_by_user.keys()), dtype=torch.long)
    if warm_users.numel() == 0:
        raise ValueError("No warm training users found in warm_emb.csv")

    targets = target_user_types(data_dir)
    if not targets:
        raise ValueError("No strict-cold/warm-up target users found.")
    target_users = sorted(targets)
    if args.limit_users and args.limit_users > 0:
        target_users = target_users[: args.limit_users]

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    warm_users_cpu = warm_users
    warm_emb = user_emb[warm_users_cpu].to(device)
    rows = []
    short_users = []
    neighbor_examples: dict[str, list[int]] = {}
    source_pos_by_user = {int(user): idx for idx, user in enumerate(warm_users_cpu.tolist())}
    take = min(args.neighbor_k, int(warm_users_cpu.numel()))

    for start in tqdm(range(0, len(target_users), args.sim_batch_size), desc="Selecting content-neighbor candidates"):
        batch_users = target_users[start : start + args.sim_batch_size]
        batch_tensor = torch.as_tensor(batch_users, dtype=torch.long)
        batch_emb = user_emb[batch_tensor].to(device)
        sims = batch_emb @ warm_emb.t()
        for row_idx, user in enumerate(batch_users):
            same_pos = source_pos_by_user.get(int(user))
            if same_pos is not None:
                sims[row_idx, same_pos] = -torch.inf
        values, local_indices = torch.topk(sims, k=take, dim=1)
        values = values.detach().cpu().numpy()
        local_indices = local_indices.detach().cpu().numpy()

        for row_idx, user in enumerate(batch_users):
            blocked = blocked_by_user.get(int(user), set())
            scores: dict[int, float] = defaultdict(float)
            neighbors = []
            for sim, local_idx in zip(values[row_idx].tolist(), local_indices[row_idx].tolist()):
                if not np.isfinite(sim):
                    continue
                warm_user = int(warm_users_cpu[int(local_idx)])
                neighbors.append(warm_user)
                contribution = max(float(sim), 0.0)
                for item in items_by_user.get(warm_user, []):
                    if item in blocked:
                        continue
                    if args.score_agg == "max":
                        scores[item] = max(scores[item], contribution)
                    else:
                        scores[item] += contribution
            ranked = sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))[: args.top_m]
            selected = [(int(item), float(score)) for item, score in ranked]
            if len(selected) < args.top_m:
                selected_items = {item for item, _ in selected}
                for item in popular_items:
                    if item in blocked or item in selected_items:
                        continue
                    selected.append((int(item), 0.0))
                    selected_items.add(int(item))
                    if len(selected) >= args.top_m:
                        break
            if len(neighbor_examples) < 5:
                neighbor_examples[str(user)] = neighbors[:10]
            if len(selected) < args.top_m:
                short_users.append({"user": int(user), "entity_type": targets[int(user)], "candidates": len(selected)})
            for item, score in selected[: args.top_m]:
                rows.append(
                    {
                        "user": int(user),
                        "item": int(item),
                        "entity_type": targets[int(user)],
                        "candidate_score": float(score),
                    }
                )

    out = pd.DataFrame(rows, columns=["user", "item", "entity_type", "candidate_score"])
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output_csv, index=False)

    meta = {
        "method": "Book-Crossing user-content-neighbor top-M",
        "definition": (
            "For each strict-cold/warm-up user, find nearest warm training users by cosine "
            "similarity in raw Book-Crossing user content, aggregate those warm users' warm "
            "training items, and select top-M items. This candidate pool uses no LoRA, no "
            "pairwise LLM scoring, and no auxiliary hybrid score."
        ),
        "dataset": data_dir.name,
        "seed": args.seed,
        "top_m": args.top_m,
        "neighbor_k": args.neighbor_k,
        "score_agg": args.score_agg,
        "device": device,
        "sim_batch_size": args.sim_batch_size,
        "train_interactions": int(len(train_df)),
        "warm_source_users": int(warm_users_cpu.numel()),
        "target_users": int(len(target_user_types(data_dir))),
        "limited_target_users": int(len(target_users)),
        "strict_rows": int((out["entity_type"] == "strict_cold").sum()) if len(out) else 0,
        "warmup_rows": int((out["entity_type"] == "warmup").sum()) if len(out) else 0,
        "rows": int(len(out)),
        "users": int(out["user"].nunique()) if len(out) else 0,
        "items": int(out["item"].nunique()) if len(out) else 0,
        "short_users": short_users[:20],
        "short_user_count": len(short_users),
        "neighbor_examples": neighbor_examples,
        "user_embedding": str(emb_path),
        "candidate_hit": candidate_hit_stats(out, data_dir),
        "output_csv": str(args.output_csv),
    }
    meta_path = args.meta_json or args.output_csv.with_suffix(".meta.json")
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2), flush=True)


if __name__ == "__main__":
    main()
