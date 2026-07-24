#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build cold-user pseudo interactions by validation-calibrated TextEmb reranking."
    )
    parser.add_argument("--candidate_csv", type=Path, required=True)
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--output_csv", type=Path, required=True)
    parser.add_argument("--summary_json", type=Path, required=True)
    parser.add_argument("--strict_k", type=int, default=10)
    parser.add_argument("--warmup_k", type=int, default=20)
    parser.add_argument("--mix_lambda", type=float, default=0.05)
    parser.add_argument(
        "--probability_mode",
        choices=["content_norm", "rerank_score", "logreg_prob"],
        default="content_norm",
    )
    parser.add_argument("--content_score_column", default="candidate_score")
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--neg_multiplier", type=int, default=30)
    parser.add_argument("--min_negatives", type=int, default=50000)
    return parser.parse_args()


def add_features(df: pd.DataFrame, data_dir: Path, score_col: str) -> tuple[pd.DataFrame, list[str]]:
    out = df.copy()
    out["_input_order"] = np.arange(len(out), dtype=np.int64)
    out["content_score"] = out[score_col].astype("float32")
    out = out.sort_values(
        ["entity_type", "user", "content_score", "_input_order"],
        ascending=[True, True, False, True],
    ).reset_index(drop=True)
    group = out.groupby(["entity_type", "user"], sort=False)["content_score"]
    out["_content_rank"] = group.cumcount() + 1
    min_score = group.transform("min").astype("float32")
    max_score = group.transform("max").astype("float32")
    mean_score = group.transform("mean").astype("float32")
    std_score = group.transform("std").fillna(0).astype("float32")
    span = (max_score - min_score).replace(0, np.nan)

    fallback_rank = 1.0 - (out["_content_rank"] - 1) / 49.0
    out["content_score_norm"] = (
        ((out["content_score"] - min_score) / span).fillna(fallback_rank).clip(0.0, 1.0).astype("float32")
    )
    out["score_to_top"] = (
        (out["content_score"] / max_score.replace(0, np.nan)).fillna(0.0).astype("float32")
    )
    out["score_z"] = (
        ((out["content_score"] - mean_score) / std_score.replace(0, np.nan))
        .fillna(0.0)
        .clip(-10.0, 10.0)
        .astype("float32")
    )
    out["rank_frac"] = ((out["_content_rank"] - 1) / 49.0).astype("float32")
    out["inv_rank"] = (1.0 / out["_content_rank"]).astype("float32")

    prev_score = group.shift(1)
    next_score = group.shift(-1)
    out["gap_prev"] = ((prev_score - out["content_score"]) / span).fillna(0.0).clip(0.0, 10.0).astype("float32")
    out["gap_next"] = ((out["content_score"] - next_score) / span).fillna(0.0).clip(0.0, 10.0).astype("float32")
    out["top_gap"] = ((max_score - out["content_score"]) / span).fillna(0.0).clip(0.0, 10.0).astype("float32")

    train = pd.read_csv(data_dir / "warm_emb.csv")
    degree = train.groupby("item").size()
    log_degree = np.log1p(out["item"].map(degree).fillna(0).astype("float32"))
    out["pop_norm"] = (log_degree / np.log1p(degree.max())).clip(0.0, 1.0).astype("float32")

    features = [
        "content_score_norm",
        "rank_frac",
        "inv_rank",
        "score_to_top",
        "score_z",
        "gap_prev",
        "gap_next",
        "top_gap",
        "pop_norm",
    ]
    return out, features


def add_validation_labels(df: pd.DataFrame, data_dir: Path) -> pd.DataFrame:
    labels = []
    for entity_type, filename in [("strict_cold", "cold_user_val.csv"), ("warmup", "warmup_val.csv")]:
        truth = pd.read_csv(data_dir / filename)[["user", "item"]].drop_duplicates()
        truth["entity_type"] = entity_type
        truth["val_label"] = 1
        labels.append(truth)
    label_df = pd.concat(labels, ignore_index=True)
    out = df.merge(label_df, on=["entity_type", "user", "item"], how="left")
    out["val_label"] = out["val_label"].fillna(0).astype("int8")
    return out


def train_predict_logreg(df: pd.DataFrame, features: list[str], args: argparse.Namespace) -> tuple[pd.DataFrame, dict]:
    rng = np.random.default_rng(args.random_seed)
    out = df.copy()
    out["logreg_prob"] = np.nan
    diagnostics = {}
    for entity_type in ["strict_cold", "warmup"]:
        sub = out[out["entity_type"] == entity_type]
        pos_idx = sub.index[sub["val_label"] == 1].to_numpy()
        neg_idx = sub.index[sub["val_label"] == 0].to_numpy()
        neg_count = min(len(neg_idx), max(args.min_negatives, len(pos_idx) * args.neg_multiplier))
        sampled_neg = rng.choice(neg_idx, size=neg_count, replace=False)
        train_idx = np.concatenate([pos_idx, sampled_neg])
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=500, class_weight="balanced", solver="lbfgs"),
        )
        clf.fit(out.loc[train_idx, features].to_numpy("float32"), out.loc[train_idx, "val_label"].to_numpy())
        out.loc[sub.index, "logreg_prob"] = clf.predict_proba(sub[features].to_numpy("float32"))[:, 1]
        diagnostics[entity_type] = {
            "validation_positive_candidates": int(len(pos_idx)),
            "sampled_negative_candidates": int(neg_count),
        }
    return out, diagnostics


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.mix_lambda <= 1.0:
        raise ValueError(f"mix_lambda must be in [0, 1], got {args.mix_lambda}")

    candidates = pd.read_csv(args.candidate_csv)
    required = {"user", "item", "entity_type", args.content_score_column}
    missing = required - set(candidates.columns)
    if missing:
        raise ValueError(f"Missing columns in {args.candidate_csv}: {sorted(missing)}")

    df, features = add_features(candidates, args.data_dir, args.content_score_column)
    df = add_validation_labels(df, args.data_dir)
    df, diagnostics = train_predict_logreg(df, features, args)
    df["rerank_score"] = (
        (1.0 - args.mix_lambda) * df["content_score_norm"] + args.mix_lambda * df["logreg_prob"]
    ).clip(0.0, 1.0).astype("float32")

    selected = []
    budgets = {"strict_cold": args.strict_k, "warmup": args.warmup_k}
    for entity_type, group in df.groupby("entity_type", sort=False):
        k = budgets.get(str(entity_type), args.warmup_k)
        chosen = (
            group.sort_values(
                ["user", "rerank_score", "content_score_norm", "_input_order"],
                ascending=[True, False, False, True],
            )
            .groupby("user", sort=False)
            .head(k)
            .copy()
        )
        selected.append(chosen)
    out = pd.concat(selected, ignore_index=True)
    if args.probability_mode == "content_norm":
        out["probability"] = out["content_score_norm"]
    elif args.probability_mode == "rerank_score":
        out["probability"] = out["rerank_score"]
    else:
        out["probability"] = out["logreg_prob"]

    out = out[
        [
            "user",
            "item",
            "entity_type",
            "probability",
            "rerank_score",
            "logreg_prob",
            "content_score_norm",
            "content_score",
            "_content_rank",
        ]
    ].sort_values(["entity_type", "user", "rerank_score"], ascending=[True, True, False])

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output_csv, index=False)
    summary = {
        "candidate_csv": str(args.candidate_csv),
        "data_dir": str(args.data_dir),
        "output_csv": str(args.output_csv),
        "strict_k": args.strict_k,
        "warmup_k": args.warmup_k,
        "mix_lambda": args.mix_lambda,
        "probability_mode": args.probability_mode,
        "features": features,
        "diagnostics": diagnostics,
        "candidate_rows": int(len(candidates)),
        "selected_rows": int(len(out)),
        "selected_users": int(out["user"].nunique()),
        "selected_items": int(out["item"].nunique()),
        "entity_type_counts": out["entity_type"].value_counts().to_dict(),
        "probability": out["probability"].describe().to_dict(),
        "rerank_score": out["rerank_score"].describe().to_dict(),
        "logreg_prob": out["logreg_prob"].describe().to_dict(),
        "content_score_norm": out["content_score_norm"].describe().to_dict(),
    }
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
