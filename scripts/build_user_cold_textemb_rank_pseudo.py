#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build cold-user pseudo interactions from TextEmb content rank."
    )
    parser.add_argument("--candidate_csv", type=Path, required=True)
    parser.add_argument("--output_csv", type=Path, required=True)
    parser.add_argument("--summary_json", type=Path, required=True)
    parser.add_argument("--strict_k", type=int, default=10)
    parser.add_argument("--warmup_k", type=int, default=20)
    parser.add_argument(
        "--weight_mode",
        choices=["content_norm", "rank_linear", "rank_inv", "rank_exp"],
        default="content_norm",
    )
    parser.add_argument("--rank_exp_gamma", type=float, default=0.85)
    parser.add_argument("--min_weight", type=float, default=0.05)
    parser.add_argument("--content_score_column", default="candidate_score")
    return parser.parse_args()


def normalize_by_user_entity(df: pd.DataFrame, score_col: str) -> pd.Series:
    pieces = []
    for _, group in df.groupby(["entity_type", "user"], sort=False):
        scores = group[score_col].astype(float)
        min_score = float(scores.min())
        max_score = float(scores.max())
        if max_score > min_score:
            norm = (scores - min_score) / (max_score - min_score)
        else:
            n = len(group)
            if n <= 1:
                norm = pd.Series([1.0] * n, index=group.index)
            else:
                ranks = pd.Series(range(n), index=group.index, dtype=float)
                norm = 1.0 - ranks / float(n - 1)
        pieces.append(norm)
    return pd.concat(pieces).sort_index() if pieces else pd.Series(dtype=float)


def rank_weight(ranks: pd.Series, k: int, mode: str, min_weight: float, gamma: float) -> pd.Series:
    r = ranks.astype(float)
    if mode == "rank_inv":
        weight = 1.0 / r
    elif mode == "rank_exp":
        weight = np.power(gamma, r - 1.0)
    elif mode == "rank_linear":
        if k <= 1:
            weight = pd.Series(1.0, index=ranks.index)
        else:
            weight = 1.0 - (r - 1.0) / float(k - 1)
    else:
        raise ValueError(f"Unexpected rank weight mode: {mode}")
    return pd.Series(weight, index=ranks.index).clip(lower=min_weight, upper=1.0)


def main() -> None:
    args = parse_args()
    candidates = pd.read_csv(args.candidate_csv)
    required = {"user", "item", "entity_type", args.content_score_column}
    missing = required - set(candidates.columns)
    if missing:
        raise ValueError(f"Missing columns in {args.candidate_csv}: {sorted(missing)}")

    df = candidates.copy()
    df["_input_order"] = range(len(df))
    df["content_score"] = df[args.content_score_column].astype(float)
    df = df.sort_values(["entity_type", "user", "content_score", "_input_order"], ascending=[True, True, False, True])
    df["_content_rank"] = df.groupby(["entity_type", "user"], sort=False).cumcount() + 1
    df["content_score_norm"] = normalize_by_user_entity(df, "content_score").clip(0.0, 1.0)

    budgets = {"strict_cold": args.strict_k, "warmup": args.warmup_k}
    selected = []
    for entity_type, group in df.groupby("entity_type", sort=False):
        k = budgets.get(str(entity_type), args.warmup_k)
        chosen = group[group["_content_rank"] <= k].copy()
        if args.weight_mode == "content_norm":
            chosen["probability"] = chosen["content_score_norm"].clip(lower=args.min_weight, upper=1.0)
        else:
            chosen["probability"] = rank_weight(
                chosen["_content_rank"],
                k=k,
                mode=args.weight_mode,
                min_weight=args.min_weight,
                gamma=args.rank_exp_gamma,
            )
        selected.append(chosen)

    out = pd.concat(selected, ignore_index=True) if selected else df.iloc[0:0].copy()
    out = out[
        [
            "user",
            "item",
            "entity_type",
            "probability",
            "content_score_norm",
            "content_score",
            "_content_rank",
        ]
    ].sort_values(["entity_type", "user", "_content_rank"], ascending=[True, True, True])

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output_csv, index=False)
    summary = {
        "candidate_csv": str(args.candidate_csv),
        "output_csv": str(args.output_csv),
        "strict_k": args.strict_k,
        "warmup_k": args.warmup_k,
        "weight_mode": args.weight_mode,
        "rank_exp_gamma": args.rank_exp_gamma,
        "min_weight": args.min_weight,
        "candidate_rows": int(len(candidates)),
        "selected_rows": int(len(out)),
        "selected_users": int(out["user"].nunique()) if len(out) else 0,
        "selected_items": int(out["item"].nunique()) if len(out) else 0,
        "entity_type_counts": out["entity_type"].value_counts().to_dict() if len(out) else {},
        "content_rank": out["_content_rank"].describe().to_dict() if len(out) else {},
        "probability": out["probability"].describe().to_dict() if len(out) else {},
        "content_score_norm": out["content_score_norm"].describe().to_dict() if len(out) else {},
    }
    args.summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
