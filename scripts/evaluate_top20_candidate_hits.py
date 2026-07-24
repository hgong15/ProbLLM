import argparse
import json
from pathlib import Path

import pandas as pd


def load_pairs(path: Path) -> set[tuple[int, int]]:
    df = pd.read_csv(path)
    return set(zip(df["user"].astype(int), df["item"].astype(int)))


def describe_top20(top20: pd.DataFrame, entity_type: str | None = None) -> dict:
    if entity_type is not None and "entity_type" in top20.columns:
        frame = top20[top20["entity_type"].eq(entity_type)]
    else:
        frame = top20
    return {
        "rows": int(len(frame)),
        "users": int(frame["user"].nunique()) if len(frame) else 0,
        "items": int(frame["item"].nunique()) if len(frame) else 0,
    }


def hit_stats(top20: pd.DataFrame, gt_pairs: set[tuple[int, int]], entity_type: str | None) -> dict:
    if entity_type is not None and "entity_type" in top20.columns:
        pred = top20[top20["entity_type"].eq(entity_type)]
    else:
        pred = top20
    pred_pairs = set(zip(pred["user"].astype(int), pred["item"].astype(int)))
    hits = len(pred_pairs & gt_pairs)
    total = len(gt_pairs)
    return {
        **describe_top20(top20, entity_type),
        "hit": int(hits),
        "total": int(total),
        "rate": float(hits / total) if total else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate top20 candidate pair hit rates.")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--top20_csv", required=True)
    parser.add_argument("--output_json", default=None)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    top20 = pd.read_csv(args.top20_csv)

    strict_gt = load_pairs(data_dir / "cold_item_test.csv")
    warmup_gt = load_pairs(data_dir / "warmup_test.csv")
    result = {
        "top20_csv": str(Path(args.top20_csv).resolve()),
        "data_dir": str(data_dir.resolve()),
        "overall": describe_top20(top20),
        "strict_cold": hit_stats(top20, strict_gt, "strict_cold"),
        "warmup": hit_stats(top20, warmup_gt, "warmup"),
    }

    text = json.dumps(result, indent=2, ensure_ascii=False)
    print(text)
    if args.output_json:
        output = Path(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
