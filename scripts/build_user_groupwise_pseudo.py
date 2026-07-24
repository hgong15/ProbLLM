#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build user-cold pseudo interactions by taking each entity group from a chosen CSV."
    )
    parser.add_argument("--strict_csv", type=Path, required=True)
    parser.add_argument("--warmup_csv", type=Path, required=True)
    parser.add_argument("--output_csv", type=Path, required=True)
    parser.add_argument("--summary_json", type=Path, required=True)
    parser.add_argument("--strict_scale", type=float, default=1.0)
    parser.add_argument("--warmup_scale", type=float, default=1.0)
    return parser.parse_args()


def read_group(path: Path, entity_type: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"user", "item", "entity_type", "probability"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {path}: {sorted(missing)}")
    out = df[df["entity_type"].astype(str) == entity_type].copy()
    if out.empty:
        raise ValueError(f"No {entity_type} rows in {path}")
    return out


def main() -> None:
    args = parse_args()
    strict = read_group(args.strict_csv, "strict_cold")
    warmup = read_group(args.warmup_csv, "warmup")
    strict["probability"] = strict["probability"].astype(float).mul(args.strict_scale).clip(0.0, 1.0)
    warmup["probability"] = warmup["probability"].astype(float).mul(args.warmup_scale).clip(0.0, 1.0)
    out = pd.concat([strict, warmup], ignore_index=True)
    out = out.drop_duplicates(subset=["user", "item", "entity_type"], keep="first")
    out = out.sort_values(["entity_type", "user", "probability"], ascending=[True, True, False])

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output_csv, index=False)

    summary = {
        "strict_csv": str(args.strict_csv.resolve()),
        "warmup_csv": str(args.warmup_csv.resolve()),
        "output_csv": str(args.output_csv.resolve()),
        "rows": int(len(out)),
        "unique_users": int(out["user"].nunique()),
        "unique_items": int(out["item"].nunique()),
        "entity_type_counts": out["entity_type"].value_counts().to_dict(),
        "strict_rows": int(len(strict)),
        "warmup_rows": int(len(warmup)),
        "strict_scale": float(args.strict_scale),
        "warmup_scale": float(args.warmup_scale),
        "probability": out["probability"].describe().to_dict(),
        "per_user_count": out.groupby("user").size().describe().to_dict(),
    }
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
