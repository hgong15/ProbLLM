#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build cold-user pseudo interactions with TextEmb rank and LLM filtering."
    )
    parser.add_argument("--candidate_csv", type=Path, required=True)
    parser.add_argument("--pred_jsonl", type=Path, required=True)
    parser.add_argument("--output_csv", type=Path, required=True)
    parser.add_argument("--summary_json", type=Path, required=True)
    parser.add_argument("--mode", choices=["high_first", "tail_swap"], required=True)
    parser.add_argument("--llm_threshold", type=float, default=0.7)
    parser.add_argument("--strict_k", type=int, default=10)
    parser.add_argument("--warmup_k", type=int, default=20)
    parser.add_argument("--strict_protect", type=int, default=5)
    parser.add_argument("--warmup_protect", type=int, default=10)
    parser.add_argument("--content_score_column", default="candidate_score")
    return parser.parse_args()


def parse_probability(value) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        prob = float(value)
    else:
        text = str(value).strip()
        try:
            prob = float(text)
        except ValueError:
            matches = re.findall(r"(?<!\d)(?:0(?:\.\d+)?|1(?:\.0+)?)(?!\d)", text)
            if not matches:
                return None
            prob = float(matches[0])
    if 0.0 <= prob <= 1.0:
        return prob
    return None


def read_predictions(path: Path) -> tuple[list[float | None], int]:
    predictions: list[float | None] = []
    invalid = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                value = json.loads(line).get("predict")
            except json.JSONDecodeError:
                value = None
            prob = parse_probability(value)
            if prob is None:
                invalid += 1
            predictions.append(prob)
    return predictions, invalid


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


def select_high_first(group: pd.DataFrame, k: int, threshold: float) -> pd.DataFrame:
    high = group[group["llm_probability"] >= threshold].sort_values("_content_rank")
    low = group[group["llm_probability"] < threshold].sort_values("_content_rank")
    return pd.concat([high, low], ignore_index=False).head(k)


def select_tail_swap(group: pd.DataFrame, k: int, threshold: float, protect: int) -> pd.DataFrame:
    ranked = group.sort_values("_content_rank")
    selected = ranked.head(k).copy()
    protected = selected.head(min(protect, k))
    tail = selected.iloc[min(protect, k):]
    pool = ranked.iloc[k:]

    replacements = pool[pool["llm_probability"] >= threshold].sort_values("_content_rank")
    if replacements.empty or tail.empty:
        return selected

    kept_tail = []
    replacement_iter = iter(replacements.index.tolist())
    used_replacements: set[int] = set()
    for idx, row in tail.iterrows():
        if row["llm_probability"] >= threshold:
            kept_tail.append(idx)
            continue
        try:
            repl_idx = next(replacement_iter)
            while repl_idx in used_replacements:
                repl_idx = next(replacement_iter)
            kept_tail.append(repl_idx)
            used_replacements.add(repl_idx)
        except StopIteration:
            kept_tail.append(idx)
    return pd.concat([protected, ranked.loc[kept_tail]], ignore_index=False).head(k)


def main() -> None:
    args = parse_args()
    candidates = pd.read_csv(args.candidate_csv)
    required = {"user", "item", "entity_type", args.content_score_column}
    missing = required - set(candidates.columns)
    if missing:
        raise ValueError(f"Missing columns in {args.candidate_csv}: {sorted(missing)}")

    predictions, invalid = read_predictions(args.pred_jsonl)
    if len(candidates) != len(predictions):
        raise ValueError(
            f"Length mismatch: candidates={len(candidates)} predictions={len(predictions)}"
        )

    df = candidates.copy()
    df["_input_order"] = range(len(df))
    df["content_score"] = df[args.content_score_column].astype(float)
    df["llm_probability"] = predictions
    df = df[df["llm_probability"].notna()].copy()
    df = df.sort_values(["entity_type", "user", "content_score", "_input_order"], ascending=[True, True, False, True])
    df["_content_rank"] = df.groupby(["entity_type", "user"], sort=False).cumcount() + 1
    df["content_score_norm"] = normalize_by_user_entity(df, "content_score").clip(0.0, 1.0)

    budgets = {"strict_cold": args.strict_k, "warmup": args.warmup_k}
    protects = {"strict_cold": args.strict_protect, "warmup": args.warmup_protect}
    selected = []
    for entity_type, entity_group in df.groupby("entity_type", sort=False):
        k = budgets.get(str(entity_type), args.warmup_k)
        protect = protects.get(str(entity_type), args.warmup_protect)
        for _, user_group in entity_group.groupby("user", sort=False):
            if args.mode == "high_first":
                chosen = select_high_first(user_group, k, args.llm_threshold)
            else:
                chosen = select_tail_swap(user_group, k, args.llm_threshold, protect)
            selected.append(chosen)

    out = pd.concat(selected, ignore_index=True) if selected else df.iloc[0:0].copy()
    out["probability"] = out["content_score_norm"]
    out = out[
        [
            "user",
            "item",
            "entity_type",
            "probability",
            "content_score_norm",
            "content_score",
            "llm_probability",
            "_content_rank",
        ]
    ].sort_values(["entity_type", "user", "_content_rank"], ascending=[True, True, True])

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output_csv, index=False)

    summary = {
        "candidate_csv": str(args.candidate_csv),
        "pred_jsonl": str(args.pred_jsonl),
        "output_csv": str(args.output_csv),
        "mode": args.mode,
        "llm_threshold": args.llm_threshold,
        "strict_k": args.strict_k,
        "warmup_k": args.warmup_k,
        "strict_protect": args.strict_protect,
        "warmup_protect": args.warmup_protect,
        "candidate_rows": int(len(candidates)),
        "prediction_rows": int(len(predictions)),
        "invalid_predictions": int(invalid),
        "usable_rows": int(len(df)),
        "selected_rows": int(len(out)),
        "selected_users": int(out["user"].nunique()) if len(out) else 0,
        "selected_items": int(out["item"].nunique()) if len(out) else 0,
        "entity_type_counts": out["entity_type"].value_counts().to_dict() if len(out) else {},
        "llm_selected_counts": out["llm_probability"].value_counts().sort_index().to_dict() if len(out) else {},
        "content_rank": out["_content_rank"].describe().to_dict() if len(out) else {},
        "content_score_norm": out["content_score_norm"].describe().to_dict() if len(out) else {},
    }
    args.summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
