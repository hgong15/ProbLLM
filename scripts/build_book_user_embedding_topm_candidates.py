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
        description="Build Book-Crossing cold-user candidates from arbitrary user/item embeddings."
    )
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--user_emb", type=Path, required=True)
    parser.add_argument("--item_emb", type=Path, required=True)
    parser.add_argument("--output_csv", type=Path, required=True)
    parser.add_argument("--meta_json", type=Path, default=None)
    parser.add_argument("--method_name", default="embedding_topm")
    parser.add_argument("--candidate_policy", default="")
    parser.add_argument("--top_m", type=int, default=20)
    parser.add_argument("--score_batch_size", type=int, default=256)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--normalize", action="store_true")
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


def load_emb(path: Path) -> torch.Tensor:
    if path.suffix == ".pt":
        emb = torch.load(path, map_location="cpu").float()
    elif path.suffix == ".npy":
        emb = torch.from_numpy(np.asarray(np.load(path, mmap_mode="r"), dtype=np.float32))
    else:
        raise ValueError(f"Unsupported embedding format: {path}")
    if emb.ndim != 2:
        raise ValueError(f"Expected 2D embedding at {path}, got shape={tuple(emb.shape)}")
    return emb.float()


def target_user_types(data_dir: Path) -> dict[int, str]:
    targets: dict[int, str] = {}
    for name in ("cold_user_val.csv", "cold_user_test.csv"):
        for user in read_pairs(data_dir / name)["user"].unique().tolist():
            targets[int(user)] = "strict_cold"
    for name in ("warmup_val.csv", "warmup_test.csv"):
        for user in read_pairs(data_dir / name)["user"].unique().tolist():
            targets[int(user)] = "warmup"
    return targets


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


def candidate_hit_stats(top20: pd.DataFrame, data_dir: Path) -> dict[str, dict[str, float | int]]:
    stats: dict[str, dict[str, float | int]] = {}
    cand_by_user = top20.groupby("user")["item"].agg(set).to_dict() if len(top20) else {}
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
        gt_users: set[int] = set()
        user_hit: set[int] = set()
        hits = 0
        for user, item in gt[["user", "item"]].itertuples(index=False):
            user_id = int(user)
            item_id = int(item)
            gt_users.add(user_id)
            if user_id in cand_users and item_id in cand_by_user.get(user_id, set()):
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
    user_emb = load_emb(args.user_emb)
    item_emb = load_emb(args.item_emb)
    if args.normalize:
        user_emb = F.normalize(user_emb, p=2, dim=1)
        item_emb = F.normalize(item_emb, p=2, dim=1)

    targets = target_user_types(data_dir)
    if not targets:
        raise ValueError("No strict-cold/warm-up target users found.")
    target_users = sorted(targets)
    if args.limit_users and args.limit_users > 0:
        target_users = target_users[: args.limit_users]
    max_user = max(target_users)
    if max_user >= user_emb.shape[0]:
        raise ValueError(f"user_emb has {user_emb.shape[0]} rows but target user id {max_user} exists.")

    train_df = read_pairs(data_dir / "warm_emb.csv", empty_ok=False)
    visible_items = sorted(train_df["item"].unique().tolist())
    if visible_items and max(visible_items) >= item_emb.shape[0]:
        raise ValueError(f"item_emb has {item_emb.shape[0]} rows but visible item id {max(visible_items)} exists.")
    blocked_by_user = observed_by_user(data_dir)

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    visible_tensor_cpu = torch.as_tensor(visible_items, dtype=torch.long)
    item_visible = item_emb[visible_tensor_cpu].to(device)

    rows = []
    for start in tqdm(range(0, len(target_users), args.score_batch_size), desc="Selecting embedding candidates"):
        batch_users = target_users[start : start + args.score_batch_size]
        batch_tensor = torch.as_tensor(batch_users, dtype=torch.long)
        scores = user_emb[batch_tensor].to(device) @ item_visible.t()
        take = min(max(args.top_m + 200, args.top_m * 20), len(visible_items))
        values, local_indices = torch.topk(scores, k=take, dim=1)
        values = values.detach().cpu().numpy()
        local_indices = local_indices.detach().cpu().numpy()
        for row_idx, user in enumerate(batch_users):
            blocked = blocked_by_user.get(int(user), set())
            selected = []
            selected_items = set()
            for score, local_idx in zip(values[row_idx].tolist(), local_indices[row_idx].tolist()):
                item = int(visible_items[int(local_idx)])
                if item in blocked or item in selected_items:
                    continue
                selected.append((item, float(score)))
                selected_items.add(item)
                if len(selected) >= args.top_m:
                    break
            if len(selected) < args.top_m:
                ranked_all = torch.argsort(scores[row_idx], descending=True).detach().cpu().tolist()
                for local_idx in ranked_all:
                    item = int(visible_items[int(local_idx)])
                    if item in blocked or item in selected_items:
                        continue
                    selected.append((item, float(scores[row_idx, int(local_idx)].detach().cpu())))
                    selected_items.add(item)
                    if len(selected) >= args.top_m:
                        break
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
        "method": args.method_name,
        "candidate_policy": args.candidate_policy
        or "Rank visible warm-training items by target-user embedding dot item embedding.",
        "dataset": data_dir.name,
        "seed": args.seed,
        "top_m": args.top_m,
        "score_batch_size": args.score_batch_size,
        "device": device,
        "normalize": bool(args.normalize),
        "user_emb": str(args.user_emb),
        "item_emb": str(args.item_emb),
        "train_interactions": int(len(train_df)),
        "visible_items": int(len(visible_items)),
        "target_users": int(len(target_user_types(data_dir))),
        "limited_target_users": int(len(target_users)),
        "rows": int(len(out)),
        "users": int(out["user"].nunique()) if len(out) else 0,
        "items": int(out["item"].nunique()) if len(out) else 0,
        "entity_type_counts": out["entity_type"].value_counts().to_dict() if len(out) else {},
        "candidate_hit": candidate_hit_stats(out, data_dir),
        "output_csv": str(args.output_csv),
    }
    meta_path = args.meta_json or args.output_csv.with_suffix(".meta.json")
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2), flush=True)


if __name__ == "__main__":
    main()
