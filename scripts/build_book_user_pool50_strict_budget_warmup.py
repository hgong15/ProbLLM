#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


DEFAULT_ROOT = Path(".")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Book-Crossing user pseudo interactions with pool50 RRF strict-cold rows and budget RRF warmup rows."
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output_csv", type=Path, required=True)
    parser.add_argument("--summary_json", type=Path, required=True)
    return parser.parse_args()


def normalize(df: pd.DataFrame, source: str) -> pd.DataFrame:
    out = df.copy()
    missing = {"user", "item", "entity_type"} - set(out.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if "probability" not in out.columns:
        out["probability"] = 1.0
    if "candidate_score" not in out.columns:
        out["candidate_score"] = out["probability"]
    out["source"] = source
    return out[["user", "item", "entity_type", "probability", "candidate_score", "source"]]


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    fusion_dir = (
        root
        / "experiments/llm_outputs/book-crossing_user_textemb_candidate_search_5seed_rrf"
        / f"seed_{args.seed}_fusion_variants"
    )
    pool_csv = fusion_dir / "rrf_pool50_c60_lam1.csv"
    budget_csv = fusion_dir / "rrf_budget_c60_lam1.csv"
    if not pool_csv.exists():
        raise FileNotFoundError(pool_csv)
    if not budget_csv.exists():
        raise FileNotFoundError(budget_csv)

    pool = pd.read_csv(pool_csv)
    budget = pd.read_csv(budget_csv)
    strict = normalize(pool[pool["entity_type"].astype(str) == "strict_cold"], "rrf_pool50_strict")
    warmup = normalize(budget[budget["entity_type"].astype(str) == "warmup"], "rrf_budget_warmup")
    mixed = pd.concat([strict, warmup], ignore_index=True)
    mixed = mixed.sort_values(["entity_type", "user"], kind="mergesort").reset_index(drop=True)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    mixed.to_csv(args.output_csv, index=False)
    summary = {
        "seed": args.seed,
        "pool_csv": str(pool_csv),
        "budget_csv": str(budget_csv),
        "output_csv": str(args.output_csv),
        "rows": int(len(mixed)),
        "users": int(mixed["user"].nunique()),
        "items": int(mixed["item"].nunique()),
        "entity_type_counts": {str(k): int(v) for k, v in mixed["entity_type"].value_counts().to_dict().items()},
        "source_counts": {str(k): int(v) for k, v in mixed["source"].value_counts().to_dict().items()},
    }
    args.summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
