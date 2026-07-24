#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Budget-control user-cold LLM pseudo interactions by per-user top-k.")
    parser.add_argument("--input_csv", type=Path, required=True)
    parser.add_argument("--output_csv", type=Path, required=True)
    parser.add_argument("--summary_json", type=Path, required=True)
    parser.add_argument("--strict_k", type=int, default=5)
    parser.add_argument("--warmup_k", type=int, default=3)
    parser.add_argument("--prob_column", default="probability")
    parser.add_argument("--min_prob", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input_csv)
    required = {"user", "item", "entity_type", args.prob_column}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {args.input_csv}: {sorted(missing)}")
    df = df[df[args.prob_column] >= args.min_prob].copy()
    if df.empty:
        raise ValueError("No rows left after probability filtering")
    df["_rank_order"] = range(len(df))
    budgets = {"strict_cold": args.strict_k, "warmup": args.warmup_k}
    pieces = []
    for entity_type, group in df.groupby("entity_type", sort=False):
        k = budgets.get(str(entity_type), args.warmup_k)
        selected = (
            group.sort_values(["user", args.prob_column, "_rank_order"], ascending=[True, False, True])
            .groupby("user", sort=False)
            .head(k)
        )
        pieces.append(selected)
    out = pd.concat(pieces, ignore_index=True).drop(columns=["_rank_order"])
    out = out.sort_values(["entity_type", "user", args.prob_column], ascending=[True, True, False])
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output_csv, index=False)
    summary = {
        "input_csv": str(args.input_csv),
        "output_csv": str(args.output_csv),
        "prob_column": args.prob_column,
        "min_prob": args.min_prob,
        "strict_k": args.strict_k,
        "warmup_k": args.warmup_k,
        "input_rows": int(len(df)),
        "output_rows": int(len(out)),
        "input_users": int(df["user"].nunique()),
        "output_users": int(out["user"].nunique()),
        "input_entity_counts": df["entity_type"].value_counts().to_dict(),
        "output_entity_counts": out["entity_type"].value_counts().to_dict(),
        "per_user_count": out.groupby("user").size().describe().to_dict(),
        "probability": out[args.prob_column].describe().to_dict(),
    }
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
