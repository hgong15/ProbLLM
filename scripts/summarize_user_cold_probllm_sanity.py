#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize one user-cold ProbLLM sanity run.")
    parser.add_argument("--run_dir", type=Path, required=True)
    parser.add_argument("--data_dir", type=Path, default=Path("data/ml-1m"))
    parser.add_argument("--candidate_json", type=Path, required=True)
    parser.add_argument("--prediction_csv", type=Path, required=True)
    parser.add_argument("--final_metrics", type=Path, required=True)
    parser.add_argument("--output_json", type=Path, default=None)
    parser.add_argument("--output_csv", type=Path, default=None)
    return parser.parse_args()


def count_csv(path: Path) -> dict:
    if not path.exists():
        return {"exists": False, "rows": 0, "users": 0, "items": 0, "entity_type_counts": {}}
    df = pd.read_csv(path)
    return {
        "exists": True,
        "rows": int(len(df)),
        "users": int(df["user"].nunique()) if "user" in df else 0,
        "items": int(df["item"].nunique()) if "item" in df else 0,
        "entity_type_counts": df["entity_type"].value_counts().to_dict() if "entity_type" in df else {},
        "probability": df["probability"].describe().to_dict() if "probability" in df and len(df) else {},
    }


def main() -> None:
    args = parse_args()
    args.run_dir.mkdir(parents=True, exist_ok=True)
    output_json = args.output_json or args.run_dir / "sanity_summary.json"
    output_csv = args.output_csv or args.run_dir / "sanity_metrics_long.csv"

    candidate = json.loads(args.candidate_json.read_text(encoding="utf-8"))
    final = json.loads(args.final_metrics.read_text(encoding="utf-8"))
    prediction_summary = count_csv(args.prediction_csv)
    summary = {
        "candidate_diagnostics": candidate,
        "pseudo_interactions": prediction_summary,
        "final_metrics": final.get("test", {}),
        "final_metrics_source": str(args.final_metrics),
    }
    output_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    rows = []
    for split, metrics in final.get("test", {}).items():
        for metric, value in metrics.items():
            rows.append({"section": "final_metrics", "split": split, "metric": metric, "value": value})
    for split, metrics in candidate.get("candidate_hit", {}).items():
        for metric, value in metrics.items():
            rows.append({"section": "candidate_hit", "split": split, "metric": metric, "value": value})
    for metric, value in prediction_summary.items():
        if isinstance(value, (int, float, str, bool)):
            rows.append({"section": "pseudo_interactions", "split": "", "metric": metric, "value": value})
    pd.DataFrame(rows).to_csv(output_csv, index=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
