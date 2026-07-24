#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build cold-user pseudo interactions by mixing content-neighbor scores "
            "with cached LLM pair probabilities."
        )
    )
    parser.add_argument("--candidate_csv", type=Path, required=True)
    parser.add_argument("--pred_jsonl", type=Path, required=True)
    parser.add_argument("--output_csv", type=Path, required=True)
    parser.add_argument("--summary_json", type=Path, required=True)
    parser.add_argument("--alpha", type=float, required=True, help="Weight for normalized content score.")
    parser.add_argument("--strict_k", type=int, default=10)
    parser.add_argument("--warmup_k", type=int, default=20)
    parser.add_argument("--content_score_column", default="candidate_score")
    parser.add_argument("--min_score", type=float, default=0.0)
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
    if prob < 0.0 or prob > 1.0:
        return None
    return prob


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


def normalize_by_user(df: pd.DataFrame, score_col: str) -> pd.Series:
    pieces = []
    for _, group in df.groupby("user", sort=False):
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
                # Candidate files are already content-ranked; use rank as a stable fallback
                # when score aggregation creates ties.
                ranks = pd.Series(range(n), index=group.index, dtype=float)
                norm = 1.0 - ranks / float(n - 1)
        pieces.append(norm)
    if not pieces:
        return pd.Series(dtype=float)
    return pd.concat(pieces).sort_index()


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {args.alpha}")

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
    df["llm_probability"] = predictions
    df = df[df["llm_probability"].notna()].copy()
    df["content_score"] = df[args.content_score_column].astype(float)
    df["content_score_norm"] = normalize_by_user(df, "content_score").clip(0.0, 1.0)
    df["hybrid_score"] = (
        args.alpha * df["content_score_norm"] + (1.0 - args.alpha) * df["llm_probability"].astype(float)
    ).clip(0.0, 1.0)
    df = df[df["hybrid_score"] >= args.min_score].copy()

    budgets = {"strict_cold": args.strict_k, "warmup": args.warmup_k}
    selected = []
    for entity_type, group in df.groupby("entity_type", sort=False):
        k = budgets.get(str(entity_type), args.warmup_k)
        chosen = (
            group.sort_values(
                ["user", "hybrid_score", "content_score_norm", "llm_probability", "_input_order"],
                ascending=[True, False, False, False, True],
            )
            .groupby("user", sort=False)
            .head(k)
        )
        selected.append(chosen)
    out = pd.concat(selected, ignore_index=True) if selected else df.iloc[0:0].copy()
    out["probability"] = out["hybrid_score"]
    out = out[
        [
            "user",
            "item",
            "entity_type",
            "probability",
            "hybrid_score",
            "llm_probability",
            "content_score_norm",
            "content_score",
        ]
    ].sort_values(["entity_type", "user", "hybrid_score"], ascending=[True, True, False])

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output_csv, index=False)

    summary = {
        "candidate_csv": str(args.candidate_csv),
        "pred_jsonl": str(args.pred_jsonl),
        "output_csv": str(args.output_csv),
        "alpha": args.alpha,
        "strict_k": args.strict_k,
        "warmup_k": args.warmup_k,
        "min_score": args.min_score,
        "candidate_rows": int(len(candidates)),
        "prediction_rows": int(len(predictions)),
        "invalid_predictions": int(invalid),
        "usable_rows": int(len(df)),
        "selected_rows": int(len(out)),
        "selected_users": int(out["user"].nunique()) if len(out) else 0,
        "selected_items": int(out["item"].nunique()) if len(out) else 0,
        "entity_type_counts": out["entity_type"].value_counts().to_dict() if len(out) else {},
        "per_user_count": out.groupby("user").size().describe().to_dict() if len(out) else {},
        "hybrid_score": out["hybrid_score"].describe().to_dict() if len(out) else {},
        "llm_probability": out["llm_probability"].describe().to_dict() if len(out) else {},
        "content_score_norm": out["content_score_norm"].describe().to_dict() if len(out) else {},
    }
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
