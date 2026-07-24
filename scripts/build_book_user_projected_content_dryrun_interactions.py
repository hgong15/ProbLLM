#!/usr/bin/env python
import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from tqdm import tqdm


def read_pairs(path, empty_ok=False):
    path = Path(path)
    if not path.exists():
        if empty_ok:
            return pd.DataFrame(columns=["user", "item"])
        raise FileNotFoundError(path)
    df = pd.read_csv(path, usecols=["user", "item"])
    return df.astype({"user": int, "item": int})


def unique_users(path, label):
    df = read_pairs(path, empty_ok=True)
    return {int(user): label for user in df["user"].unique().tolist()}


def normalize_rows(x):
    x = np.asarray(x, dtype=np.float32)
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norm, 1e-12)


def minmax(values):
    values = np.asarray(values, dtype=np.float32)
    if len(values) == 0:
        return values
    lo = float(values.min())
    hi = float(values.max())
    if hi - lo < 1e-12:
        return np.ones_like(values, dtype=np.float32) * 0.5
    return (values - lo) / (hi - lo)


def hit_stats(candidates, truth_path, label, target_users=None):
    truth = read_pairs(truth_path, empty_ok=True)
    if target_users is not None:
        truth = truth[truth["user"].isin(target_users)]
    if len(truth) == 0:
        return {"hits": 0, "total": 0, "hit_rate": 0.0}
    cand = candidates[candidates["entity_type"] == label]
    cand_pairs = set(zip(cand["user"].astype(int), cand["item"].astype(int)))
    hits = sum((int(user), int(item)) in cand_pairs for user, item in truth[["user", "item"]].itertuples(index=False))
    return {"hits": int(hits), "total": int(len(truth)), "hit_rate": float(hits / max(len(truth), 1))}


def project_item_content(item_content, item_ids, proj_dim, seed, batch_size):
    rng = np.random.RandomState(seed)
    random_matrix = rng.normal(
        loc=0.0,
        scale=1.0 / np.sqrt(proj_dim),
        size=(item_content.shape[1], proj_dim),
    ).astype(np.float32)
    out = np.empty((len(item_ids), proj_dim), dtype=np.float32)
    for start in tqdm(range(0, len(item_ids), batch_size), desc="Projecting item content"):
        end = min(start + batch_size, len(item_ids))
        dense = np.asarray(item_content[item_ids[start:end]], dtype=np.float32)
        item_sparse = sparse.csr_matrix(dense)
        out[start:end] = item_sparse @ random_matrix
    return normalize_rows(out)


def build_user_profiles(train_df, item_proj, item_row_by_id, user_num, proj_dim):
    users = train_df["user"].to_numpy(dtype=np.int64)
    items = train_df["item"].to_numpy(dtype=np.int64)
    rows = item_row_by_id[items]
    valid = rows >= 0
    users = users[valid]
    rows = rows[valid]

    profiles = np.zeros((user_num, proj_dim), dtype=np.float32)
    counts = np.zeros(user_num, dtype=np.float32)
    np.add.at(profiles, users, item_proj[rows])
    np.add.at(counts, users, 1.0)
    active = counts > 0
    profiles[active] /= counts[active, None]
    return normalize_rows(profiles), counts


def fit_ridge_projection(user_content, profiles, counts, ridge, max_fit_users, seed):
    fit_users = np.flatnonzero(counts > 0)
    if max_fit_users and len(fit_users) > max_fit_users:
        rng = np.random.RandomState(seed)
        fit_users = rng.choice(fit_users, size=max_fit_users, replace=False)
        fit_users.sort()

    x = np.asarray(user_content[fit_users], dtype=np.float32)
    y = profiles[fit_users].astype(np.float32)
    bias = np.ones((x.shape[0], 1), dtype=np.float32)
    x_aug = np.concatenate([x, bias], axis=1)

    xtx = x_aug.T @ x_aug
    reg = np.eye(xtx.shape[0], dtype=np.float32) * float(ridge)
    reg[-1, -1] = 0.0
    xty = x_aug.T @ y
    weights = np.linalg.solve(xtx + reg, xty).astype(np.float32)
    pred = normalize_rows(x_aug @ weights)
    target = normalize_rows(y)
    fit_cos = np.sum(pred * target, axis=1)
    return weights, {
        "fit_users": int(len(fit_users)),
        "fit_cos_mean": float(fit_cos.mean()) if len(fit_cos) else 0.0,
        "fit_cos_std": float(fit_cos.std()) if len(fit_cos) else 0.0,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Dry-run Book-Crossing user-side pseudo interactions via content-to-item-space projection."
    )
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--diagnostics_json", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--limit_users", type=int, default=2000)
    parser.add_argument("--proj_dim", type=int, default=128)
    parser.add_argument("--ridge", type=float, default=10.0)
    parser.add_argument("--max_fit_users", type=int, default=50000)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--score_batch_size", type=int, default=256)
    args = parser.parse_args()

    rng = np.random.RandomState(args.seed)
    data_dir = Path(args.data_dir)
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with (data_dir / "convert_dict.pkl").open("rb") as f:
        para = pickle.load(f)
    if para.get("cold_object") != "user":
        raise ValueError(f"Expected cold_object=user in convert_dict.pkl, got {para.get('cold_object')!r}")

    targets = {}
    for name in ("cold_user_val.csv", "cold_user_test.csv"):
        targets.update(unique_users(data_dir / name, "strict_cold"))
    for name in ("warmup_val.csv", "warmup_test.csv"):
        targets.update(unique_users(data_dir / name, "warmup"))
    target_users = np.asarray(sorted(targets), dtype=np.int64)
    if args.limit_users and args.limit_users > 0:
        target_users = target_users[: args.limit_users]
    if len(target_users) == 0:
        raise ValueError("No strict-cold/warm-up users found.")

    train_df = read_pairs(data_dir / "warm_emb.csv", empty_ok=False)
    visible_items = np.asarray(sorted(train_df["item"].unique().tolist()), dtype=np.int64)
    train_items_by_user = train_df.groupby("user")["item"].agg(set).to_dict()

    user_content = np.load(data_dir / "book-crossing_user_content.npy", mmap_mode="r")
    item_content = np.load(data_dir / "book-crossing_item_content.npy", mmap_mode="r")
    user_num = int(para["user_num"])
    item_num = int(para["item_num"])
    if user_content.shape[0] < user_num or item_content.shape[0] < item_num:
        raise ValueError(
            f"Content shape mismatch: user_content={user_content.shape}, item_content={item_content.shape}, "
            f"expected at least ({user_num}, {item_num})"
        )

    item_proj = project_item_content(item_content, visible_items, args.proj_dim, args.seed, args.batch_size)
    item_row_by_id = np.full(item_num, -1, dtype=np.int64)
    item_row_by_id[visible_items] = np.arange(len(visible_items), dtype=np.int64)
    profiles, counts = build_user_profiles(train_df, item_proj, item_row_by_id, user_num, args.proj_dim)
    weights, fit_diag = fit_ridge_projection(
        user_content,
        profiles,
        counts,
        ridge=args.ridge,
        max_fit_users=args.max_fit_users,
        seed=args.seed,
    )

    rows = []
    item_scores_matrix = item_proj.T
    for start in tqdm(range(0, len(target_users), args.score_batch_size), desc="Selecting Book user candidates"):
        end = min(start + args.score_batch_size, len(target_users))
        batch_users = target_users[start:end]
        x = np.asarray(user_content[batch_users], dtype=np.float32)
        x_aug = np.concatenate([x, np.ones((x.shape[0], 1), dtype=np.float32)], axis=1)
        user_repr = normalize_rows(x_aug @ weights)
        scores = user_repr @ item_scores_matrix

        for row_idx, user in enumerate(batch_users.tolist()):
            blocked = train_items_by_user.get(int(user), set())
            score_row = scores[row_idx]
            take = min(len(score_row), max(args.topk * 20, args.topk + 200))
            if take < len(score_row):
                idx = np.argpartition(-score_row, take - 1)[:take]
            else:
                idx = np.arange(len(score_row))
            idx = idx[np.argsort(-score_row[idx])]

            selected = []
            selected_scores = []
            for local_idx in idx.tolist():
                item = int(visible_items[local_idx])
                if item in blocked:
                    continue
                selected.append(item)
                selected_scores.append(float(score_row[local_idx]))
                if len(selected) >= args.topk:
                    break
            if len(selected) < args.topk:
                fallback = visible_items.copy()
                rng.shuffle(fallback)
                for item in fallback.tolist():
                    item = int(item)
                    if item in blocked or item in selected:
                        continue
                    local_idx = int(item_row_by_id[item])
                    selected.append(item)
                    selected_scores.append(float(score_row[local_idx]))
                    if len(selected) >= args.topk:
                        break

            probs = 0.2 + 0.7 * minmax(np.asarray(selected_scores, dtype=np.float32))
            for item, prob in zip(selected, probs.tolist()):
                rows.append(
                    {
                        "user": int(user),
                        "item": int(item),
                        "entity_type": targets[int(user)],
                        "probability": float(prob),
                    }
                )

    out = pd.DataFrame(rows, columns=["user", "item", "entity_type", "probability"])
    out.to_csv(output_csv, index=False)

    diagnostics = {
        "dataset": "book-crossing",
        "cold_object": "user",
        "seed": args.seed,
        "method": "dryrun ridge-projected user content to random-projected item content top-k",
        "topk": args.topk,
        "limit_users": args.limit_users,
        "proj_dim": args.proj_dim,
        "ridge": args.ridge,
        "max_fit_users": args.max_fit_users,
        "target_users": int(len(target_users)),
        "all_target_users": int(len(targets)),
        "rows": int(len(out)),
        "unique_users": int(out["user"].nunique()) if len(out) else 0,
        "unique_items": int(out["item"].nunique()) if len(out) else 0,
        "entity_type_counts": out["entity_type"].value_counts().to_dict() if len(out) else {},
        "visible_items": int(len(visible_items)),
        "fit_diagnostics": fit_diag,
        "strict_candidate_hit": hit_stats(out, data_dir / "cold_user_test.csv", "strict_cold", set(target_users.tolist())),
        "warmup_candidate_hit": hit_stats(out, data_dir / "warmup_test.csv", "warmup", set(target_users.tolist())),
        "output_csv": str(output_csv),
    }
    print(json.dumps(diagnostics, indent=2, ensure_ascii=False))
    if args.diagnostics_json:
        Path(args.diagnostics_json).write_text(json.dumps(diagnostics, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
