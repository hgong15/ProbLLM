#!/usr/bin/env python
import argparse
import json
from pathlib import Path

import pandas as pd


def read_dat(path, names):
    return pd.read_csv(
        path,
        sep="::",
        names=names,
        engine="python",
        encoding="latin1",
    )


def main():
    parser = argparse.ArgumentParser(description="Prepare MovieLens-1M files for the ProbLLM revision protocol.")
    parser.add_argument("--data_dir", default="data/ml-1m")
    parser.add_argument("--min_rating", type=float, default=4.0)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    ratings = read_dat(data_dir / "ratings.dat", ["raw_user", "raw_item", "rating", "timestamp"])
    movies = read_dat(data_dir / "movies.dat", ["raw_item", "title", "genres"])

    positives = ratings[ratings["rating"] >= args.min_rating].copy()
    raw_users = sorted(positives["raw_user"].unique().tolist())
    raw_items = sorted(positives["raw_item"].unique().tolist())
    user_map = {raw_id: idx for idx, raw_id in enumerate(raw_users)}
    item_map = {raw_id: idx for idx, raw_id in enumerate(raw_items)}

    interactions = positives[["raw_user", "raw_item"]].copy()
    interactions["user"] = interactions["raw_user"].map(user_map).astype(int)
    interactions["item"] = interactions["raw_item"].map(item_map).astype(int)
    interactions = interactions[["user", "item"]].drop_duplicates().sort_values(["user", "item"])

    item_content = (
        movies[movies["raw_item"].isin(raw_items)]
        .copy()
        .assign(item=lambda df: df["raw_item"].map(item_map).astype(int))
        .sort_values("item")
    )
    item_content["title"] = item_content["title"].astype(str)
    item_content["genres"] = item_content["genres"].astype(str)
    item_content = item_content[["item", "raw_item", "title", "genres"]]

    interactions.to_csv(data_dir / "ml-1m.csv", index=False)
    item_content.to_csv(data_dir / "raw-data.csv", index=False)

    metadata = {
        "source": "MovieLens 1M",
        "min_rating_as_positive": args.min_rating,
        "users": len(raw_users),
        "items": len(raw_items),
        "interactions": len(interactions),
        "raw_user_min_max": [int(min(raw_users)), int(max(raw_users))],
        "raw_item_count_in_movies_dat": int(len(movies)),
    }
    (data_dir / "preprocess_meta.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
