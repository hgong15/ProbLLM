import argparse
from pathlib import Path

import pandas as pd


def read_csv(path):
    frame = pd.read_csv(path)
    if "probability" not in frame.columns:
        frame["probability"] = 1.0
    if "entity_type" not in frame.columns:
        frame["entity_type"] = ""
    return frame[["user", "item", "entity_type", "probability"]].copy()


def main():
    parser = argparse.ArgumentParser(description="Merge pseudo-interaction CSV files and keep max probability.")
    parser.add_argument("--input_csv", action="append", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--summary_txt", default=None)
    args = parser.parse_args()

    frames = [read_csv(path) for path in args.input_csv]
    merged = pd.concat(frames, ignore_index=True)
    merged["user"] = merged["user"].astype(int)
    merged["item"] = merged["item"].astype(int)
    merged["probability"] = pd.to_numeric(merged["probability"], errors="coerce").fillna(1.0)
    merged["probability"] = merged["probability"].clip(0.0, 1.0)
    merged = (
        merged.sort_values(["user", "item", "entity_type", "probability"], ascending=[True, True, True, False])
        .drop_duplicates(["user", "item", "entity_type"], keep="first")
        .sort_values(["item", "entity_type", "user"])
        .reset_index(drop=True)
    )

    out = Path(args.output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out, index=False)
    summary = [
        f"inputs={args.input_csv}",
        f"rows={len(merged)}",
        f"users={merged['user'].nunique()}",
        f"items={merged['item'].nunique()}",
        "entity_rows=" + repr(merged["entity_type"].value_counts().to_dict()),
        "probability=" + repr(merged["probability"].describe().to_dict()),
        f"output={out.resolve()}",
    ]
    text = "\n".join(summary)
    print(text)
    if args.summary_txt:
        Path(args.summary_txt).parent.mkdir(parents=True, exist_ok=True)
        Path(args.summary_txt).write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
