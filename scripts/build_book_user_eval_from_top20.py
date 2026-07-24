#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Book-Crossing cold-user LLM eval JSON from top-k candidates.")
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--top20_csv", type=Path, required=True)
    parser.add_argument("--output_json", type=Path, required=True)
    parser.add_argument("--max_history", type=int, default=20)
    parser.add_argument("--metadata_csv", type=Path, default=None)
    parser.add_argument("--title_column", default=None)
    parser.add_argument("--item_id_column", default=None)
    return parser.parse_args()


def load_titles(args: argparse.Namespace) -> dict[int, str]:
    if args.metadata_csv is None or not args.metadata_csv.exists():
        return {}
    meta = pd.read_csv(args.metadata_csv)
    title_column = args.title_column
    if title_column is None:
        for candidate in ("title", "book_title", "Book-Title", "BookTitle", "name"):
            if candidate in meta.columns:
                title_column = candidate
                break
    item_id_column = args.item_id_column
    if item_id_column is None:
        for candidate in ("item", "item_id", "book_id", "Book-ID", "ISBN"):
            if candidate in meta.columns:
                item_id_column = candidate
                break
    if title_column is None or item_id_column is None:
        raise ValueError(
            f"Could not infer title/item columns from {args.metadata_csv}. "
            "Pass --title_column and --item_id_column explicitly."
        )
    titles = {}
    for item_id, title in zip(meta[item_id_column], meta[title_column]):
        try:
            key = int(item_id)
        except (TypeError, ValueError):
            continue
        title_text = str(title).strip()
        if title_text:
            titles[key] = title_text
    return titles


def book_text(item_id: int, titles: dict[int, str]) -> str:
    title = titles.get(item_id)
    if title:
        return title
    return f"Book {item_id}"


def load_histories(data_dir: Path, titles: dict[int, str], max_history: int) -> dict[int, str]:
    grouped: dict[int, list[str]] = defaultdict(list)
    seen: dict[int, set[int]] = defaultdict(set)
    for warm_path in (data_dir / "warm_emb.csv", data_dir / "warmup_support.csv"):
        if not warm_path.exists() and warm_path.name == "warm_emb.csv":
            warm_path = data_dir / "warm_train.csv"
        if not warm_path.exists():
            continue
        train = pd.read_csv(warm_path, usecols=["user", "item"])
        for user, item in zip(train["user"].astype(int), train["item"].astype(int)):
            user_id = int(user)
            item_id = int(item)
            if item_id in seen[user_id]:
                continue
            seen[user_id].add(item_id)
            values = grouped[user_id]
            if len(values) < max_history:
                values.append(book_text(item_id, titles))
    return {user: ", ".join(f'"{value}"' for value in values) for user, values in grouped.items()}


def main() -> None:
    args = parse_args()
    titles = load_titles(args)
    histories = load_histories(args.data_dir, titles, args.max_history)
    top20 = pd.read_csv(args.top20_csv)
    required = {"user", "item"}
    missing = required - set(top20.columns)
    if missing:
        raise ValueError(f"{args.top20_csv} missing columns: {sorted(missing)}")

    instruction = (
        "Given the user's historical book set, predict the probability "
        "(a value between 0 and 1, e.g., 0.85) that the user will like the target book."
    )
    examples = []
    for row in top20.itertuples(index=False):
        user_id = int(row.user)
        item_id = int(row.item)
        entity_type = getattr(row, "entity_type", "")
        history = histories.get(user_id, "")
        if not history:
            history = '"No visible history"'
        examples.append(
            {
                "instruction": instruction,
                "input": (
                    f"User preference: {history}, "
                    f'What is the probability the user will like the target book "{book_text(item_id, titles)}"?'
                ),
                "output": "1",
                "user_id": user_id,
                "item_id": item_id,
                "entity_type": entity_type,
            }
        )

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(examples, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        f"Saved {len(examples)} Book-Crossing LLM eval examples to {args.output_json}; "
        f"titles={len(titles)}, users_with_history={len(histories)}"
    )


if __name__ == "__main__":
    main()
