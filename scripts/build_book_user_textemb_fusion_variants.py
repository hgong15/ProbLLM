#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build conservative TextEmb fusion pseudo-interaction variants for Book-Crossing user cold start."
    )
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--original_budget_csv", type=Path, required=True)
    parser.add_argument("--new_budget_csv", type=Path, required=True)
    parser.add_argument("--original_pool_csv", type=Path, default=None)
    parser.add_argument("--new_pool_csv", type=Path, default=None)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--strict_k", type=int, default=10)
    parser.add_argument("--warmup_k", type=int, default=20)
    parser.add_argument("--rrf_c", type=float, default=60.0)
    parser.add_argument("--rrf_lambda", type=float, default=1.0)
    return parser.parse_args()


def read_pairs(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["user", "item"])
    df = pd.read_csv(path)
    if df.empty:
        return pd.DataFrame(columns=["user", "item"])
    return df[["user", "item"]].astype({"user": int, "item": int})


def truth_by_split(data_dir: Path) -> dict[str, dict[int, set[int]]]:
    files = {
        "strict_val": "cold_user_val.csv",
        "strict_test": "cold_user_test.csv",
        "warmup_val": "warmup_val.csv",
        "warmup_test": "warmup_test.csv",
    }
    out: dict[str, dict[int, set[int]]] = {}
    for split, name in files.items():
        by_user: dict[int, set[int]] = {}
        for user, group in read_pairs(data_dir / name).groupby("user", sort=False):
            by_user[int(user)] = set(group["item"].astype(int).tolist())
        out[split] = by_user
    return out


def load_ranked(path: Path, source: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "probability" not in df.columns:
        if "content_score_norm" in df.columns:
            df["probability"] = df["content_score_norm"].astype(float)
        elif "candidate_score" in df.columns:
            df["probability"] = normalize_by_user(df, "candidate_score")
        else:
            df["probability"] = 1.0
    if "candidate_score" not in df.columns:
        if "content_score" in df.columns:
            df["candidate_score"] = df["content_score"].astype(float)
        elif "hybrid_score" in df.columns:
            df["candidate_score"] = df["hybrid_score"].astype(float)
        else:
            df["candidate_score"] = df["probability"].astype(float)
    df = df[["user", "item", "entity_type", "probability", "candidate_score"]].copy()
    df["user"] = df["user"].astype(int)
    df["item"] = df["item"].astype(int)
    df["probability"] = df["probability"].astype(float).clip(0.0, 1.0)
    df["candidate_score"] = df["candidate_score"].astype(float)
    df["_input_order"] = np.arange(len(df))
    df = df.sort_values(
        ["entity_type", "user", "candidate_score", "probability", "_input_order"],
        ascending=[True, True, False, False, True],
        kind="mergesort",
    )
    df["rank"] = df.groupby(["entity_type", "user"], sort=False).cumcount() + 1
    df["source"] = source
    return df


def normalize_by_user(df: pd.DataFrame, score_col: str) -> pd.Series:
    parts = []
    for _, group in df.groupby(["entity_type", "user"], sort=False):
        values = group[score_col].astype(float)
        lo = float(values.min())
        hi = float(values.max())
        if hi > lo:
            norm = (values - lo) / (hi - lo)
        else:
            norm = pd.Series([1.0] * len(group), index=group.index)
        parts.append(norm)
    return pd.concat(parts).sort_index() if parts else pd.Series(dtype=float)


def group_map(df: pd.DataFrame) -> dict[tuple[str, int], list[dict]]:
    out = {}
    for key, group in df.groupby(["entity_type", "user"], sort=False):
        out[(str(key[0]), int(key[1]))] = group.to_dict("records")
    return out


def budget_for(entity_type: str, strict_k: int, warmup_k: int) -> int:
    return strict_k if entity_type == "strict_cold" else warmup_k


def add_unique(selected: list[dict], seen: set[int], rows: list[dict], limit: int) -> None:
    for row in rows:
        item = int(row["item"])
        if item in seen:
            continue
        selected.append(row)
        seen.add(item)
        if len(selected) >= limit:
            break


def apply_gamma(df: pd.DataFrame, gamma: float) -> pd.DataFrame:
    out = df.copy()
    out["probability"] = out["probability"].astype(float).clip(0.0, 1.0) ** float(gamma)
    return out


def rows_to_df(rows: list[dict], gamma: float | None = None) -> pd.DataFrame:
    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=["user", "item", "entity_type", "probability", "candidate_score", "source"])
    out = out[["user", "item", "entity_type", "probability", "candidate_score", "source"]].copy()
    out["probability"] = out["probability"].astype(float).clip(0.0, 1.0)
    if gamma is not None:
        out["probability"] = out["probability"] ** float(gamma)
    return out.sort_values(["entity_type", "user"], kind="mergesort").reset_index(drop=True)


def build_core_tail(
    original: dict[tuple[str, int], list[dict]],
    new: dict[tuple[str, int], list[dict]],
    keys: list[tuple[str, int]],
    strict_orig: int,
    strict_new: int,
    warm_orig: int,
    warm_new: int,
    strict_k: int,
    warmup_k: int,
    warmup_original_only: bool = False,
    gamma: float | None = None,
) -> pd.DataFrame:
    rows = []
    for entity_type, user in keys:
        limit = budget_for(entity_type, strict_k, warmup_k)
        if entity_type == "warmup" and warmup_original_only:
            rows.extend(original.get((entity_type, user), [])[:limit])
            continue
        orig_take = strict_orig if entity_type == "strict_cold" else warm_orig
        new_take = strict_new if entity_type == "strict_cold" else warm_new
        selected: list[dict] = []
        seen: set[int] = set()
        orig_rows = original.get((entity_type, user), [])
        new_rows = new.get((entity_type, user), [])
        add_unique(selected, seen, orig_rows[:orig_take], limit)
        add_unique(selected, seen, new_rows[: max(new_take, limit)], limit)
        add_unique(selected, seen, orig_rows[orig_take:], limit)
        add_unique(selected, seen, new_rows, limit)
        rows.extend(selected[:limit])
    return rows_to_df(rows, gamma=gamma)


def build_source_mix(
    original: dict[tuple[str, int], list[dict]],
    new: dict[tuple[str, int], list[dict]],
    keys: list[tuple[str, int]],
    strict_source: str,
    warmup_source: str,
    strict_k: int,
    warmup_k: int,
    gamma: float | None = None,
) -> pd.DataFrame:
    rows = []
    for entity_type, user in keys:
        source = strict_source if entity_type == "strict_cold" else warmup_source
        mapping = new if source == "new" else original
        rows.extend(mapping.get((entity_type, user), [])[: budget_for(entity_type, strict_k, warmup_k)])
    return rows_to_df(rows, gamma=gamma)


def build_rrf(
    original: dict[tuple[str, int], list[dict]],
    new: dict[tuple[str, int], list[dict]],
    keys: list[tuple[str, int]],
    strict_k: int,
    warmup_k: int,
    c: float,
    lam: float,
    warmup_original_only: bool = False,
    gamma: float | None = None,
) -> pd.DataFrame:
    rows = []
    for entity_type, user in keys:
        limit = budget_for(entity_type, strict_k, warmup_k)
        if entity_type == "warmup" and warmup_original_only:
            rows.extend(original.get((entity_type, user), [])[:limit])
            continue
        candidates: dict[int, dict] = {}
        for weight, source_rows in [(1.0, original.get((entity_type, user), [])), (lam, new.get((entity_type, user), []))]:
            for rank0, row in enumerate(source_rows, start=1):
                item = int(row["item"])
                entry = candidates.setdefault(
                    item,
                    {
                        "user": user,
                        "item": item,
                        "entity_type": entity_type,
                        "candidate_score": 0.0,
                        "probability": 0.0,
                        "source": "rrf",
                    },
                )
                entry["candidate_score"] += float(weight) / (float(c) + float(rank0))
                entry["probability"] = max(float(entry["probability"]), float(row["probability"]))
        ranked = sorted(candidates.values(), key=lambda row: (-float(row["candidate_score"]), int(row["item"])))[:limit]
        if ranked:
            scores = np.asarray([float(row["candidate_score"]) for row in ranked], dtype=np.float32)
            lo = float(scores.min())
            hi = float(scores.max())
            if hi > lo:
                probs = (scores - lo) / (hi - lo)
            else:
                probs = np.ones_like(scores, dtype=np.float32)
            for row, prob in zip(ranked, probs.tolist()):
                row["probability"] = float(prob)
        rows.extend(ranked)
    return rows_to_df(rows, gamma=gamma)


def candidate_stats(df: pd.DataFrame, truth: dict[str, dict[int, set[int]]]) -> dict[str, float | int]:
    by_key = {
        (str(entity), int(user)): set(group["item"].astype(int).tolist())
        for (entity, user), group in df.groupby(["entity_type", "user"], sort=False)
    }
    out: dict[str, float | int] = {}
    for split, entity_type in [
        ("strict_val", "strict_cold"),
        ("strict_test", "strict_cold"),
        ("warmup_val", "warmup"),
        ("warmup_test", "warmup"),
    ]:
        hits = 0
        total = 0
        user_hit = 0
        users = 0
        for user, items in truth[split].items():
            selected = by_key.get((entity_type, int(user)), set())
            found = len(items & selected)
            hits += found
            total += len(items)
            user_hit += int(found > 0)
            users += 1
        out[f"{split}_hits"] = int(hits)
        out[f"{split}_total"] = int(total)
        out[f"{split}_recall"] = float(hits / total) if total else 0.0
        out[f"{split}_user_hit_rate"] = float(user_hit / users) if users else 0.0
    out["val_macro_recall"] = 0.5 * (float(out["strict_val_recall"]) + float(out["warmup_val_recall"]))
    out["test_macro_recall"] = 0.5 * (float(out["strict_test_recall"]) + float(out["warmup_test_recall"]))
    return out


def write_variant(name: str, df: pd.DataFrame, output_dir: Path, truth: dict[str, dict[int, set[int]]]) -> dict:
    csv_path = output_dir / f"{name}.csv"
    df[["user", "item", "entity_type", "probability", "candidate_score", "source"]].to_csv(csv_path, index=False)
    stats = candidate_stats(df, truth)
    summary = {
        "variant": name,
        "csv": str(csv_path),
        "rows": int(len(df)),
        "users": int(df["user"].nunique()) if len(df) else 0,
        "items": int(df["item"].nunique()) if len(df) else 0,
        "entity_type_counts": df["entity_type"].value_counts().to_dict() if len(df) else {},
        "probability": df["probability"].describe().to_dict() if len(df) else {},
        **stats,
    }
    csv_path.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    original_budget = load_ranked(args.original_budget_csv, "orig")
    new_budget = load_ranked(args.new_budget_csv, "new")
    original_pool = load_ranked(args.original_pool_csv, "orig_pool") if args.original_pool_csv else original_budget
    new_pool = load_ranked(args.new_pool_csv, "new_pool") if args.new_pool_csv else new_budget

    original_budget_map = group_map(original_budget)
    new_budget_map = group_map(new_budget)
    original_pool_map = group_map(original_pool)
    new_pool_map = group_map(new_pool)
    keys = sorted(set(original_budget_map) | set(new_budget_map))
    truth = truth_by_split(args.data_dir)

    variants: dict[str, pd.DataFrame] = {}
    variants["strict_new_warm_orig"] = build_source_mix(
        original_budget_map,
        new_budget_map,
        keys,
        strict_source="new",
        warmup_source="orig",
        strict_k=args.strict_k,
        warmup_k=args.warmup_k,
    )
    variants["coretail_s7n3_w15n5"] = build_core_tail(
        original_budget_map, new_budget_map, keys, 7, 3, 15, 5, args.strict_k, args.warmup_k
    )
    variants["coretail_s7n3_w20orig"] = build_core_tail(
        original_budget_map,
        new_budget_map,
        keys,
        7,
        3,
        args.warmup_k,
        0,
        args.strict_k,
        args.warmup_k,
        warmup_original_only=True,
    )
    variants["coretail_s8n2_w16n4"] = build_core_tail(
        original_budget_map, new_budget_map, keys, 8, 2, 16, 4, args.strict_k, args.warmup_k
    )
    variants["rrf_budget_c60_lam1"] = build_rrf(
        original_budget_map, new_budget_map, keys, args.strict_k, args.warmup_k, args.rrf_c, args.rrf_lambda
    )
    variants["rrf_budget_c60_lam1_warm_orig"] = build_rrf(
        original_budget_map,
        new_budget_map,
        keys,
        args.strict_k,
        args.warmup_k,
        args.rrf_c,
        args.rrf_lambda,
        warmup_original_only=True,
    )
    variants["rrf_pool50_c60_lam1"] = build_rrf(
        original_pool_map, new_pool_map, keys, args.strict_k, args.warmup_k, args.rrf_c, args.rrf_lambda
    )
    variants["rrf_pool50_c60_lam1_warm_orig"] = build_rrf(
        original_pool_map,
        new_pool_map,
        keys,
        args.strict_k,
        args.warmup_k,
        args.rrf_c,
        args.rrf_lambda,
        warmup_original_only=True,
    )
    variants["k500_rank_decay_gamma15"] = build_source_mix(
        original_budget_map, new_budget_map, keys, "new", "new", args.strict_k, args.warmup_k, gamma=1.5
    )
    variants["k500_rank_decay_gamma20"] = build_source_mix(
        original_budget_map, new_budget_map, keys, "new", "new", args.strict_k, args.warmup_k, gamma=2.0
    )
    variants["coretail_s7n3_w20orig_gamma15"] = build_core_tail(
        original_budget_map,
        new_budget_map,
        keys,
        7,
        3,
        args.warmup_k,
        0,
        args.strict_k,
        args.warmup_k,
        warmup_original_only=True,
        gamma=1.5,
    )

    summaries = [write_variant(name, df, args.output_dir, truth) for name, df in variants.items()]
    summary_df = pd.DataFrame(summaries).sort_values("val_macro_recall", ascending=False)
    summary_df.to_csv(args.output_dir / "fusion_variant_candidate_summary.csv", index=False)
    try:
        summary_df.to_excel(args.output_dir / "fusion_variant_candidate_summary.xlsx", index=False)
    except Exception:
        pass
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
