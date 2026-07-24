#!/usr/bin/env python
import argparse
import json
import pickle
from pathlib import Path

import pandas as pd


def read_pairs(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["user", "item"])
    return pd.read_csv(path, usecols=["user", "item"]).astype({"user": int, "item": int})


def load_convert(data_dir: Path) -> dict:
    path = data_dir / "convert_dict.pkl"
    if not path.exists():
        return {}
    with path.open("rb") as f:
        obj = pickle.load(f)
    return obj if isinstance(obj, dict) else {}


def infer_cold_object(data_dir: Path, requested: str | None) -> str:
    if requested and requested != "auto":
        return requested
    cold_object = load_convert(data_dir).get("cold_object")
    if cold_object in {"item", "user"}:
        return cold_object
    if (data_dir / "cold_item_test.csv").exists():
        return "item"
    if (data_dir / "cold_user_test.csv").exists():
        return "user"
    raise ValueError(f"Could not infer cold_object from {data_dir}")


def parse_candidate(value: str) -> tuple[str, Path]:
    if "=" in value:
        label, path = value.split("=", 1)
    elif ":" in value:
        label, path = value.split(":", 1)
    else:
        path = value
        label = Path(value).parent.name
    label = label.strip()
    if not label:
        raise ValueError(f"Empty candidate label in {value!r}")
    return label, Path(path)


def split_truth(data_dir: Path, cold_object: str) -> dict[str, pd.DataFrame]:
    if cold_object == "item":
        strict_file = data_dir / "cold_item_test.csv"
        if not strict_file.exists() and (data_dir / "strict_cold_item_test.csv").exists():
            strict_file = data_dir / "strict_cold_item_test.csv"
        return {
            "strict_cold": read_pairs(strict_file),
            "warmup": read_pairs(data_dir / "warmup_test.csv"),
        }
    return {
        "strict_cold": read_pairs(data_dir / "cold_user_test.csv"),
        "warmup": read_pairs(data_dir / "warmup_test.csv"),
    }


def safe_float(value: float) -> float:
    return float(value) if pd.notna(value) else 0.0


def evaluate_one(topk: pd.DataFrame, truth: pd.DataFrame, split: str, cold_object: str) -> dict:
    target_col = "item" if cold_object == "item" else "user"
    other_col = "user" if cold_object == "item" else "item"

    if "entity_type" in topk.columns:
        cand = topk[topk["entity_type"].astype(str).eq(split)].copy()
    else:
        cand = topk.copy()
    if len(cand):
        cand = cand.astype({"user": int, "item": int})
    truth = truth.astype({"user": int, "item": int})

    pred_pairs = set(zip(cand["user"].tolist(), cand["item"].tolist())) if len(cand) else set()
    truth_pairs = list(zip(truth["user"].tolist(), truth["item"].tolist()))
    hits = sum(1 for pair in truth_pairs if pair in pred_pairs)

    target_entities = sorted(truth[target_col].unique().tolist()) if len(truth) else []
    candidate_target_entities = set(cand[target_col].unique().tolist()) if len(cand) else set()
    hit_target_entities = set()
    entity_recalls = []
    if len(cand):
        cand_by_target = {
            int(target): set(group[other_col].astype(int).tolist())
            for target, group in cand.groupby(target_col, sort=False)
        }
    else:
        cand_by_target = {}

    if len(truth):
        for target, group in truth.groupby(target_col, sort=False):
            gt_other = set(group[other_col].astype(int).tolist())
            cand_other = cand_by_target.get(int(target), set())
            entity_hits = len(gt_other & cand_other)
            if entity_hits > 0:
                hit_target_entities.add(int(target))
            entity_recalls.append(entity_hits / max(len(gt_other), 1))

    return {
        "split": split,
        "candidate_rows": int(len(cand)),
        "candidate_targets": int(len(candidate_target_entities)),
        "candidate_users": int(cand["user"].nunique()) if len(cand) else 0,
        "candidate_items": int(cand["item"].nunique()) if len(cand) else 0,
        "gt_pairs": int(len(truth_pairs)),
        "gt_targets": int(len(target_entities)),
        "hits": int(hits),
        "pair_recall": float(hits / len(truth_pairs)) if truth_pairs else 0.0,
        "target_coverage": float(len(candidate_target_entities & set(target_entities)) / len(target_entities)) if target_entities else 0.0,
        "target_hit_rate": float(len(hit_target_entities) / len(target_entities)) if target_entities else 0.0,
        "mean_target_recall": safe_float(pd.Series(entity_recalls).mean()) if entity_recalls else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare top-K candidate hit rates under one cold-start split.")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--cold_object", choices=["auto", "item", "user"], default="auto")
    parser.add_argument("--candidate", action="append", required=True, help="Format: label=/path/to/top20.csv")
    parser.add_argument("--output_csv", default=None)
    parser.add_argument("--output_json", default=None)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    cold_object = infer_cold_object(data_dir, args.cold_object)
    truth_by_split = split_truth(data_dir, cold_object)
    rows = []
    details = {
        "data_dir": str(data_dir.resolve()),
        "cold_object": cold_object,
        "candidates": {},
    }
    for candidate_arg in args.candidate:
        label, path = parse_candidate(candidate_arg)
        topk = pd.read_csv(path)
        details["candidates"][label] = str(path.resolve())
        for split, truth in truth_by_split.items():
            metrics = evaluate_one(topk, truth, split, cold_object)
            rows.append({"method": label, **metrics})

    out = pd.DataFrame(rows)
    print(out.to_string(index=False))
    if args.output_csv:
        output_csv = Path(args.output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(output_csv, index=False)
    if args.output_json:
        output_json = Path(args.output_json)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        details["metrics"] = rows
        output_json.write_text(json.dumps(details, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
