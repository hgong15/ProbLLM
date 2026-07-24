import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


def read_pairs(path: Path, empty_ok: bool = True) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    if empty_ok:
        return pd.DataFrame(columns=["user", "item"])
    raise FileNotFoundError(path)


def unique_users(path: Path, entity_type: str) -> dict[int, str]:
    df = read_pairs(path)
    if df.empty:
        return {}
    return {int(user): entity_type for user in sorted(df["user"].unique().tolist())}


def minmax(values: np.ndarray) -> np.ndarray:
    lo = float(values.min())
    hi = float(values.max())
    if hi <= lo:
        return np.full_like(values, 0.5, dtype=np.float32)
    return ((values - lo) / (hi - lo)).astype(np.float32)


def hit_stats(rows: pd.DataFrame, gt_path: Path, entity_type: str) -> dict:
    gt = read_pairs(gt_path)
    pred = rows[rows["entity_type"] == entity_type]
    if gt.empty or pred.empty:
        return {"hits": 0, "total": int(len(gt)), "hit_rate": 0.0}
    gt_pairs = {(int(r.user), int(r.item)) for r in gt.itertuples(index=False)}
    pred_pairs = {(int(r.user), int(r.item)) for r in pred.itertuples(index=False)}
    hits = len(gt_pairs & pred_pairs)
    return {"hits": hits, "total": len(gt_pairs), "hit_rate": hits / max(len(gt_pairs), 1)}


def main():
    parser = argparse.ArgumentParser(
        description="Dry-run MovieLens user-side simulated interactions using frozen text embeddings."
    )
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--limit_users", type=int, default=0, help="Optional quick-test cap over target users.")
    parser.add_argument("--diagnostics_json", default=None)
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
    target_users = sorted(targets)
    if args.limit_users and args.limit_users > 0:
        target_users = target_users[: args.limit_users]
    if not target_users:
        raise ValueError("No strict-cold/warm-up users found.")

    train_df = read_pairs(data_dir / "warm_emb.csv", empty_ok=False)
    visible_items = np.asarray(sorted(train_df["item"].unique().tolist()), dtype=np.int64)
    train_items_by_user = train_df.groupby("user")["item"].agg(set).to_dict()

    user_emb = torch.load(data_dir / "llm_user_content_emb.pt", map_location="cpu").float().numpy()
    item_emb = torch.load(data_dir / "llm_item_content_emb.pt", map_location="cpu").float().numpy()
    user_emb = user_emb / np.maximum(np.linalg.norm(user_emb, axis=1, keepdims=True), 1e-12)
    item_emb = item_emb / np.maximum(np.linalg.norm(item_emb, axis=1, keepdims=True), 1e-12)

    rows = []
    candidate_pool = min(len(visible_items), max(args.topk * 10, args.topk + 100))
    item_matrix = item_emb[visible_items]

    for user in tqdm(target_users, desc="Selecting user-side text-embedding candidates"):
        scores = item_matrix @ user_emb[int(user)]
        blocked = train_items_by_user.get(int(user), set())
        take = min(candidate_pool, len(scores))
        if take < len(scores):
            idx = np.argpartition(-scores, take - 1)[:take]
        else:
            idx = np.arange(len(scores))
        idx = idx[np.argsort(-scores[idx])]

        selected = []
        selected_scores = []
        for local_idx in idx.tolist():
            item = int(visible_items[local_idx])
            if item in blocked:
                continue
            selected.append(item)
            selected_scores.append(float(scores[local_idx]))
            if len(selected) >= args.topk:
                break
        if len(selected) < args.topk:
            fallback = visible_items.copy()
            rng.shuffle(fallback)
            for item in fallback.tolist():
                item = int(item)
                if item in blocked or item in selected:
                    continue
                selected.append(item)
                selected_scores.append(float(scores[np.where(visible_items == item)[0][0]]))
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
        "dataset": "ml-1m",
        "cold_object": "user",
        "seed": args.seed,
        "method": "dryrun frozen LLM user/item text embedding top-k",
        "topk": args.topk,
        "limit_users": args.limit_users,
        "target_users": len(target_users),
        "rows": len(out),
        "unique_users": int(out["user"].nunique()) if len(out) else 0,
        "unique_items": int(out["item"].nunique()) if len(out) else 0,
        "entity_type_counts": out["entity_type"].value_counts().to_dict() if len(out) else {},
        "strict_candidate_hit": hit_stats(out, data_dir / "cold_user_test.csv", "strict_cold"),
        "warmup_candidate_hit": hit_stats(out, data_dir / "warmup_test.csv", "warmup"),
        "output_csv": str(output_csv),
    }
    print(json.dumps(diagnostics, indent=2, ensure_ascii=False))
    if args.diagnostics_json:
        Path(args.diagnostics_json).write_text(json.dumps(diagnostics, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
