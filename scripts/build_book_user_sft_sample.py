#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pickle
import random
from collections import defaultdict
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Book-Crossing user-cold SFT samples.")
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--output_json", type=Path, required=True)
    parser.add_argument("--max_examples", type=int, default=6000)
    parser.add_argument("--max_history", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_history_fraction", type=float, default=0.15)
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
        raise ValueError("Could not infer metadata title/item columns.")

    titles: dict[int, str] = {}
    for item_id, title in zip(meta[item_id_column], meta[title_column]):
        try:
            key = int(item_id)
        except (TypeError, ValueError):
            continue
        text = str(title).strip()
        if text:
            titles[key] = text
    return titles


def book_text(item_id: int, titles: dict[int, str]) -> str:
    title = titles.get(int(item_id))
    if title:
        return title
    return f"Book {int(item_id)}"


def read_positive_interactions(data_dir: Path) -> pd.DataFrame:
    pieces = []
    for name in ("warm_emb.csv", "warm_train.csv", "warmup_support.csv"):
        path = data_dir / name
        if not path.exists():
            continue
        df = pd.read_csv(path, usecols=["user", "item"]).drop_duplicates()
        pieces.append(df)
    if not pieces:
        raise FileNotFoundError(f"No warm interaction CSVs found under {data_dir}")
    return pd.concat(pieces, ignore_index=True).drop_duplicates().reset_index(drop=True)


def build_histories(pos_df: pd.DataFrame, max_history: int) -> tuple[dict[int, list[int]], dict[int, set[int]]]:
    histories: dict[int, list[int]] = defaultdict(list)
    seen: dict[int, set[int]] = defaultdict(set)
    for user, item in zip(pos_df["user"].astype(int), pos_df["item"].astype(int)):
        user_id = int(user)
        item_id = int(item)
        if item_id in seen[user_id]:
            continue
        seen[user_id].add(item_id)
        if len(histories[user_id]) < max_history:
            histories[user_id].append(item_id)
    return histories, seen


def render_history(user_id: int, target_item: int, histories: dict[int, list[int]], titles: dict[int, str]) -> str:
    items = [item for item in histories.get(int(user_id), []) if int(item) != int(target_item)]
    if not items:
        return '"No visible history"'
    return ", ".join(f'"{book_text(item, titles)}"' for item in items)


def make_example(
    user_id: int,
    item_id: int,
    label: int,
    rng: random.Random,
    histories: dict[int, list[int]],
    titles: dict[int, str],
    no_history_fraction: float,
) -> dict[str, object]:
    if rng.random() < no_history_fraction:
        history = '"No visible history"'
    else:
        history = render_history(user_id, item_id, histories, titles)
    if label == 1:
        prob = round(rng.uniform(0.6, 1.0), 2)
    else:
        prob = round(rng.uniform(0.0, 0.4), 2)
    return {
        "instruction": (
            "Given the user's historical book set, predict the probability "
            "(a value between 0 and 1, e.g., 0.85) that the user will like the target book."
        ),
        "input": (
            f"User preference: {history}, "
            f'What is the probability the user will like the target book "{book_text(item_id, titles)}"?'
        ),
        "output": str(prob),
        "user_id": int(user_id),
        "item_id": int(item_id),
    }


def sample_negative_item(user_seen: set[int], item_num: int, rng: random.Random) -> int:
    for _ in range(100):
        item_id = rng.randrange(item_num)
        if item_id not in user_seen:
            return item_id
    candidates = set(range(item_num)) - user_seen
    if not candidates:
        return rng.randrange(item_num)
    return rng.choice(tuple(candidates))


def main() -> None:
    args = parse_args()
    if args.max_examples <= 0:
        raise ValueError("--max_examples must be positive")
    rng = random.Random(args.seed)
    titles = load_titles(args)
    pos_df = read_positive_interactions(args.data_dir)
    histories, seen = build_histories(pos_df, args.max_history)

    with (args.data_dir / "n_user_item.pkl").open("rb") as handle:
        n_user_item = pickle.load(handle)
    item_num = int(n_user_item["item"])

    pos_count = args.max_examples // 2
    neg_count = args.max_examples - pos_count
    pos_sample = pos_df.sample(n=pos_count, replace=len(pos_df) < pos_count, random_state=args.seed)

    examples = []
    for row in pos_sample.itertuples(index=False):
        examples.append(
            make_example(
                int(row.user),
                int(row.item),
                1,
                rng,
                histories,
                titles,
                args.no_history_fraction,
            )
        )

    users = list(seen)
    for _ in range(neg_count):
        user_id = int(rng.choice(users))
        item_id = sample_negative_item(seen[user_id], item_num, rng)
        examples.append(
            make_example(user_id, item_id, 0, rng, histories, titles, args.no_history_fraction)
        )

    rng.shuffle(examples)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(examples, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        json.dumps(
            {
                "output_json": str(args.output_json),
                "examples": len(examples),
                "positive_examples": pos_count,
                "negative_examples": neg_count,
                "users_with_history": len(histories),
                "item_num": item_num,
                "titles": len(titles),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
