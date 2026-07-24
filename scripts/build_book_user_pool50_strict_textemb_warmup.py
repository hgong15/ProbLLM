#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


DEFAULT_ROOT = Path(".")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Book-Crossing user pseudo interactions with pool50 RRF strict-cold rows and TextEmb top10 warmup rows."
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output_csv", type=Path, required=True)
    parser.add_argument("--summary_json", type=Path, required=True)
    parser.add_argument("--strict_k", type=int, default=10)
    parser.add_argument("--warmup_k", type=int, default=10)
    return parser.parse_args()


def ensure_columns(df: pd.DataFrame, path: Path) -> None:
    missing = {"user", "item", "entity_type"} - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing required columns: {sorted(missing)}")


def normalize_sim(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "probability" not in out.columns:
        out["probability"] = 1.0
    if "candidate_score" not in out.columns:
        out["candidate_score"] = out["probability"]
    if "source" not in out.columns:
        out["source"] = "mixed"
    return out[["user", "item", "entity_type", "probability", "candidate_score", "source"]]


def textemb_topk_from_top20(path: Path, warmup_k: int) -> pd.DataFrame:
    top20 = pd.read_csv(path)
    ensure_columns(top20, path)
    sort_cols = [col for col in ["entity_type", "user"] if col in top20.columns]
    score_col = "candidate_score" if "candidate_score" in top20.columns else None
    if score_col is not None:
        top20 = top20.sort_values(
            sort_cols + [score_col],
            ascending=[True] * len(sort_cols) + [False],
            kind="mergesort",
        )
    warm = top20[top20["entity_type"].astype(str) == "warmup"].copy()
    warm = warm.groupby(["entity_type", "user"], sort=False, group_keys=False).head(warmup_k)
    warm["probability"] = 1.0
    if "candidate_score" not in warm.columns:
        warm["candidate_score"] = 1.0
    warm["source"] = "textemb_top10_warmup"
    return normalize_sim(warm)


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    seed = args.seed
    pool_csv = (
        root
        / "experiments/llm_outputs/book-crossing_user_textemb_candidate_search_5seed_rrf"
        / f"seed_{seed}_fusion_variants/rrf_pool50_c60_lam1.csv"
    )
    top20_csv = (
        root
        / "experiments/llm_outputs/book-crossing_user_content_neighbor_top20_k100_user5seed"
        / f"content_neighbor_candidate_seed_{seed}/top20.csv"
    )
    if not pool_csv.exists():
        raise FileNotFoundError(pool_csv)
    if not top20_csv.exists():
        raise FileNotFoundError(top20_csv)

    pool = pd.read_csv(pool_csv)
    ensure_columns(pool, pool_csv)
    if args.strict_k <= 0 or args.warmup_k <= 0:
        raise ValueError("strict_k and warmup_k must be positive")
    strict = pool[pool["entity_type"].astype(str) == "strict_cold"].copy()
    strict_score = "candidate_score" if "candidate_score" in strict.columns else "probability"
    strict = strict.sort_values(
        ["entity_type", "user", strict_score],
        ascending=[True, True, False],
        kind="mergesort",
    )
    strict = strict.groupby(["entity_type", "user"], sort=False, group_keys=False).head(args.strict_k)
    strict["source"] = strict.get("source", "rrf_pool50_strict")
    strict = normalize_sim(strict)
    warm = textemb_topk_from_top20(top20_csv, args.warmup_k)
    mixed = pd.concat([strict, warm], ignore_index=True)
    mixed = mixed.sort_values(["entity_type", "user"], kind="mergesort").reset_index(drop=True)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    mixed.to_csv(args.output_csv, index=False)
    summary = {
        "seed": seed,
        "pool_csv": str(pool_csv),
        "top20_csv": str(top20_csv),
        "output_csv": str(args.output_csv),
        "strict_k": args.strict_k,
        "warmup_k": args.warmup_k,
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
