#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build item-cold pseudo interactions by blending per-user candidate scores with LLM scores."
    )
    parser.add_argument("--top_csv", required=True)
    parser.add_argument("--prediction_jsonl", default=None)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--topks", default="20,30,50")
    parser.add_argument("--llm_weights", default="0.00,0.05,0.10,0.25,0.50")
    parser.add_argument(
        "--entity_topks",
        default="",
        help="Optional comma-separated overrides by entity_type, e.g. strict_cold:10,warmup:20,warm:5.",
    )
    parser.add_argument("--rank_min_probability", type=float, default=0.60)
    parser.add_argument("--rank_max_probability", type=float, default=1.00)
    parser.add_argument("--prefix", default="probllm_useritem_blend")
    return parser.parse_args()


def parse_probability(value):
    if value is None:
        return None
    if isinstance(value, (int, float)):
        prob = float(value)
    else:
        text = str(value).strip()
        try:
            prob = float(text)
        except ValueError:
            matches = re.findall(r"(?<![\d.])(?:0(?:\.\d+)?|1(?:\.0+)?)(?!\d)", text)
            if not matches:
                return None
            prob = float(matches[0])
    if math.isnan(prob) or prob < 0.0 or prob > 1.0:
        return None
    return prob


def read_predictions(path: Path | None, expected_rows: int) -> tuple[list[float | None], int]:
    if path is None:
        return [0.5] * expected_rows, 0
    predictions: list[float | None] = []
    invalid = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                predictions.append(None)
                invalid += 1
                continue
            prob = parse_probability(record.get("predict"))
            if prob is None:
                invalid += 1
            predictions.append(prob)
    return predictions, invalid


def parse_list(text: str, cast):
    return [cast(part) for part in text.split(",") if part.strip()]


def parse_entity_topks(text: str) -> dict[str, int]:
    out: dict[str, int] = {}
    if not text.strip():
        return out
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            raise ValueError(f"Invalid entity_topks entry: {part!r}")
        key, value = part.split(":", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Invalid empty entity_type in entity_topks entry: {part!r}")
        topk = int(value)
        if topk <= 0:
            raise ValueError(f"entity_topks must be positive, got {part!r}")
        out[key] = topk
    return out


def normalize(values: list[float]) -> list[float]:
    if not values:
        return []
    lo = min(values)
    hi = max(values)
    if hi <= lo:
        return [0.5 for _ in values]
    return [(value - lo) / (hi - lo) for value in values]


def tag_float(value: float) -> str:
    return f"{value:.2f}".replace(".", "p")


def write_edges(
    path: Path,
    groups,
    topk: int,
    llm_weight: float,
    rank_min: float,
    rank_max: float,
    entity_topks: dict[str, int],
) -> int:
    emb_weight = 1.0 - llm_weight
    rows = []
    for key in sorted(groups):
        entity_type = str(key[1])
        group_topk = entity_topks.get(entity_type, topk)
        group = groups[key]
        llm_norm = normalize([row["llm_probability"] for row in group])
        emb_norm = normalize([row["embedding_score"] for row in group])
        scored = []
        for row, llm_value, emb_value in zip(group, llm_norm, emb_norm):
            blend_score = llm_weight * llm_value + emb_weight * emb_value
            scored.append((blend_score, llm_value, emb_value, -int(row["item"]), row))
        scored.sort(reverse=True)
        keep = [entry[-1] for entry in scored[:group_topk]]
        denom = max(len(keep) - 1, 1)
        for idx, row in enumerate(keep):
            frac = 1.0 - idx / denom
            probability = rank_min + (rank_max - rank_min) * frac
            rows.append(
                {
                    "user": int(row["user"]),
                    "item": int(row["item"]),
                    "entity_type": row["entity_type"],
                    "probability": f"{probability:.6f}",
                }
            )

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["user", "item", "entity_type", "probability"])
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def main() -> None:
    args = parse_args()
    top_csv = Path(args.top_csv)
    prediction_jsonl = Path(args.prediction_jsonl) if args.prediction_jsonl else None
    output_dir = Path(args.output_dir)
    topks = parse_list(args.topks, int)
    llm_weights = parse_list(args.llm_weights, float)
    entity_topks = parse_entity_topks(args.entity_topks)

    candidates = pd.read_csv(top_csv)
    predictions, invalid = read_predictions(prediction_jsonl, len(candidates))
    if len(candidates) != len(predictions):
        raise ValueError(f"Length mismatch: candidates={len(candidates)} predictions={len(predictions)}")

    rows = []
    has_candidate_score = "candidate_score" in candidates.columns
    for idx, (row, prob) in enumerate(zip(candidates.to_dict("records"), predictions)):
        if prob is None:
            continue
        if has_candidate_score:
            emb_score = float(row["candidate_score"])
        else:
            rank = int(row.get("candidate_rank", idx + 1))
            emb_score = -float(rank)
        rows.append(
            {
                "user": int(row["user"]),
                "item": int(row["item"]),
                "entity_type": str(row.get("entity_type", "")),
                "llm_probability": float(prob),
                "embedding_score": emb_score,
            }
        )

    groups = defaultdict(list)
    for row in rows:
        groups[(int(row["user"]), str(row["entity_type"]))].append(row)

    outputs = []
    for topk in topks:
        for llm_weight in llm_weights:
            if llm_weight < 0.0 or llm_weight > 1.0:
                raise ValueError(f"llm_weight must be in [0,1], got {llm_weight}")
            emb_weight = 1.0 - llm_weight
            tag = f"{args.prefix}_top{topk}_llm{tag_float(llm_weight)}_emb{tag_float(emb_weight)}"
            path = output_dir / f"{tag}_rank.csv"
            row_count = write_edges(
                path,
                groups,
                topk,
                llm_weight,
                args.rank_min_probability,
                args.rank_max_probability,
                entity_topks,
            )
            outputs.append(
                {
                    "name": tag,
                    "path": str(path.resolve()),
                    "topk": int(topk),
                    "llm_weight": float(llm_weight),
                    "embedding_weight": float(emb_weight),
                    "rows": int(row_count),
                }
            )

    summary = {
        "top_csv": str(top_csv.resolve()),
        "prediction_jsonl": str(prediction_jsonl.resolve()) if prediction_jsonl else None,
        "candidate_rows": int(len(candidates)),
        "prediction_rows": int(len(predictions)),
        "valid_prediction_rows": int(len(rows)),
        "invalid_predictions": int(invalid),
        "groups": int(len(groups)),
        "grouping": "user,entity_type",
        "entity_topks": entity_topks,
        "rank_min_probability": args.rank_min_probability,
        "rank_max_probability": args.rank_max_probability,
        "outputs": outputs,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "blend_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
