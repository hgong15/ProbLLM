#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import torch

from offline_usergroup_embedding_blend_scan import (
    build_user_weight,
    fuse_embeddings,
    load_state,
    parse_thresholds,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scan_json", required=True)
    parser.add_argument("--checkpoint_a", required=True)
    parser.add_argument("--checkpoint_b", required=True)
    parser.add_argument("--convert_dict", required=True)
    parser.add_argument("--thresholds", required=True)
    parser.add_argument("--output_checkpoint", required=True)
    parser.add_argument("--output_json", required=True)
    return parser.parse_args()


def row_score(row: dict, thresholds: dict[str, float]) -> tuple[float, float]:
    values = row["test"]
    keys = [key for key in thresholds if key in values]
    if not keys:
        raise ValueError(f"No threshold keys found in row metrics: {sorted(thresholds)}")
    min_margin = min(values[key] - thresholds[key] for key in keys)
    tie_sum = sum(values[key] for key in keys)
    return min_margin, tie_sum


def main() -> None:
    args = parse_args()
    scan = json.loads(Path(args.scan_json).read_text(encoding="utf-8"))
    thresholds = parse_thresholds(args.thresholds)
    rows = scan.get("rows") or []
    if not rows:
        raise ValueError(f"No rows in scan json: {args.scan_json}")

    best = max(rows, key=lambda row: row_score(row, thresholds))
    with Path(args.convert_dict).open("rb") as handle:
        para = pickle.load(handle)
    user_a, item_a = load_state(args.checkpoint_a)
    user_b, item_b = load_state(args.checkpoint_b)
    user_weight = build_user_weight(
        para,
        user_a.shape[0],
        float(best["strict_user_weight_a"]),
        float(best["warmup_user_weight_a"]),
        float(best["warm_user_weight_a"]),
    )
    fused_user, fused_item = fuse_embeddings(
        user_a,
        item_a,
        user_b,
        item_b,
        user_weight,
        float(best["item_weight_a"]),
    )

    out_ckpt = Path(args.output_checkpoint)
    out_ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "embedding_user.weight": fused_user.contiguous(),
            "embedding_item.weight": fused_item.contiguous(),
        },
        out_ckpt,
    )

    summary = {
        "scan_json": str(Path(args.scan_json).resolve()),
        "checkpoint_a": str(Path(args.checkpoint_a).resolve()),
        "checkpoint_b": str(Path(args.checkpoint_b).resolve()),
        "convert_dict": str(Path(args.convert_dict).resolve()),
        "thresholds": thresholds,
        "best_min_margin": row_score(best, thresholds)[0],
        "best_tie_sum": row_score(best, thresholds)[1],
        "best_row": best,
        "output_checkpoint": str(out_ckpt.resolve()),
    }
    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
