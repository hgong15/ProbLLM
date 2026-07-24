#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pickle
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


AGE_MAP = {
    1: "under 18",
    18: "18-24",
    25: "25-34",
    35: "35-44",
    45: "45-49",
    50: "50-55",
    56: "56 or older",
}

OCCUPATION_MAP = {
    0: "other or not specified",
    1: "academic or educator",
    2: "artist",
    3: "clerical or admin",
    4: "college or graduate student",
    5: "customer service",
    6: "doctor or health care",
    7: "executive or managerial",
    8: "farmer",
    9: "homemaker",
    10: "K-12 student",
    11: "lawyer",
    12: "programmer",
    13: "retired",
    14: "sales or marketing",
    15: "scientist",
    16: "self-employed",
    17: "technician or engineer",
    18: "tradesman or craftsman",
    19: "unemployed",
    20: "writer",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build MovieLens cold-user side-info prompts and SFT data."
    )
    parser.add_argument("--data_dir", type=Path, default=Path("data/ml-1m"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_history", type=int, default=20)
    parser.add_argument("--negative_ratio", type=float, default=1.0)
    parser.add_argument("--user_text_output", default="user_side_context_list.pkl")
    parser.add_argument("--train_json_output", default="train_sample_user_cold.json")
    parser.add_argument("--meta_output", default="user_cold_side_data_meta.json")
    return parser.parse_args()


def read_pairs(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["user", "item"])
    df = pd.read_csv(path)
    if df.empty:
        return pd.DataFrame(columns=["user", "item"])
    return df[["user", "item"]].astype({"user": int, "item": int})


def load_user_profiles(data_dir: Path, n_user: int) -> list[str]:
    users_path = data_dir / "users.dat"
    rows = {}
    with users_path.open("r", encoding="latin1") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            user_raw, gender, age, occupation, zipcode = line.split("::")
            # MovieLens ids are 1-based in users.dat and remapped to 0-based
            # elsewhere in this repository.
            user_id = int(user_raw) - 1
            gender_text = "female" if gender == "F" else "male"
            age_text = AGE_MAP.get(int(age), str(age))
            occupation_text = OCCUPATION_MAP.get(int(occupation), str(occupation))
            rows[user_id] = (
                f"gender: {gender_text}; age: {age_text}; "
                f"occupation: {occupation_text}; zipcode: {zipcode}"
            )
    profiles = []
    for user in range(n_user):
        profiles.append(rows.get(user, "gender: unknown; age: unknown; occupation: unknown; zipcode: unknown"))
    return profiles


def load_item_titles(data_dir: Path) -> list[str]:
    item_df = pd.read_csv(data_dir / "raw-data.csv")
    return [str(title) for title in item_df["title"].tolist()]


def visible_histories(train_df: pd.DataFrame, titles: list[str], n_user: int, max_history: int) -> list[list[str]]:
    items_by_user: dict[int, list[int]] = defaultdict(list)
    for user, item in train_df[["user", "item"]].itertuples(index=False):
        items_by_user[int(user)].append(int(item))
    histories = []
    for user in range(n_user):
        item_ids = items_by_user.get(user, [])[:max_history]
        histories.append([titles[item] for item in item_ids if 0 <= item < len(titles)])
    return histories


def make_context(profile: str, history_titles: list[str]) -> str:
    if history_titles:
        history = '", "'.join(history_titles)
        return f'User profile: {profile}. Observed liked movies: "{history}"'
    return f"User profile: {profile}. Observed liked movies: none"


def sample_negative_items(item_num: int, positives: set[int], rng: np.random.Generator, n: int) -> list[int]:
    if n <= 0:
        return []
    candidates = np.asarray(sorted(set(range(item_num)) - positives), dtype=np.int64)
    if len(candidates) == 0:
        return []
    n = min(n, len(candidates))
    return rng.choice(candidates, size=n, replace=False).astype(int).tolist()


def build_train_samples(
    train_df: pd.DataFrame,
    contexts: list[str],
    titles: list[str],
    item_num: int,
    seed: int,
    negative_ratio: float,
) -> list[dict]:
    rng = np.random.default_rng(seed)
    py_rng = random.Random(seed)
    samples = []
    grouped = train_df.groupby("user")["item"].agg(list).to_dict()
    instruction = (
        "Given the user's profile and observed movie interactions, predict the probability "
        "(a value between 0 and 1, e.g., 0.32) that the user will like the target movie."
    )
    for user, pos_items in tqdm(grouped.items(), desc="Building user-cold SFT samples"):
        user = int(user)
        positives = {int(item) for item in pos_items}
        for item in pos_items:
            item = int(item)
            prob = round(py_rng.uniform(0.6, 1.0), 2)
            samples.append(
                {
                    "instruction": instruction,
                    "input": f'{contexts[user]}, target movie: "{titles[item]}".',
                    "output": str(prob),
                    "user_id": user,
                    "item_id": item,
                }
            )
        n_neg = int(round(len(pos_items) * negative_ratio))
        for item in sample_negative_items(item_num, positives, rng, n_neg):
            prob = round(py_rng.uniform(0.0, 0.4), 2)
            samples.append(
                {
                    "instruction": instruction,
                    "input": f'{contexts[user]}, target movie: "{titles[item]}".',
                    "output": str(prob),
                    "user_id": user,
                    "item_id": item,
                }
            )
    py_rng.shuffle(samples)
    return samples


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir
    with (data_dir / "n_user_item.pkl").open("rb") as handle:
        n_user_item = pickle.load(handle)
    n_user = int(n_user_item["user"])
    item_num = int(n_user_item["item"])

    train_df = read_pairs(data_dir / "warm_emb.csv")
    titles = load_item_titles(data_dir)
    profiles = load_user_profiles(data_dir, n_user)
    histories = visible_histories(train_df, titles, n_user, args.max_history)
    contexts = [make_context(profile, history) for profile, history in zip(profiles, histories)]
    samples = build_train_samples(train_df, contexts, titles, item_num, args.seed, args.negative_ratio)

    user_text_path = data_dir / args.user_text_output
    train_json_path = data_dir / args.train_json_output
    meta_path = data_dir / args.meta_output
    with user_text_path.open("wb") as handle:
        pickle.dump(contexts, handle)
    train_json_path.write_text(json.dumps(samples, indent=4, ensure_ascii=False), encoding="utf-8")

    strict_users = sorted(set(read_pairs(data_dir / "cold_user_val.csv")["user"]) | set(read_pairs(data_dir / "cold_user_test.csv")["user"]))
    warmup_users = sorted(set(read_pairs(data_dir / "warmup_val.csv")["user"]) | set(read_pairs(data_dir / "warmup_test.csv")["user"]))
    meta = {
        "dataset": "ml-1m",
        "seed": args.seed,
        "n_user": n_user,
        "item_num": item_num,
        "train_interactions": int(len(train_df)),
        "train_samples": int(len(samples)),
        "negative_ratio": args.negative_ratio,
        "max_history": args.max_history,
        "strict_cold_users": len(strict_users),
        "warmup_users": len(warmup_users),
        "strict_no_observed_history_count": int(sum("Observed liked movies: none" in contexts[user] for user in strict_users)),
        "warmup_has_support_count": int(sum("Observed liked movies: none" not in contexts[user] for user in warmup_users)),
        "user_text_output": str(user_text_path),
        "train_json_output": str(train_json_path),
        "example_strict_user_context": contexts[strict_users[0]] if strict_users else "",
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(meta, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
