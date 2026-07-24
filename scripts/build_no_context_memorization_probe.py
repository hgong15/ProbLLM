import argparse
import json
import pickle
import random
import re
from pathlib import Path

import pandas as pd


def parse_prob(value):
    if value is None:
        return None
    if isinstance(value, (float, int)):
        prob = float(value)
    else:
        text = str(value).strip()
        try:
            prob = float(text)
        except ValueError:
            match = re.search(r"(?<![\d.])(?:0\.\d+|1(?:\.0+)?|0)(?!\d)", text)
            if not match:
                return None
            prob = float(match.group(0))
    return prob if 0.0 <= prob <= 1.0 else None


def auc_from_labels(scores, labels):
    valid = [(s, l) for s, l in zip(scores, labels) if s is not None]
    if not valid:
        return None
    scores, labels = zip(*valid)
    n = len(labels)
    n_pos = sum(labels)
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    order = sorted(range(n), key=lambda idx: scores[idx])
    rank_sum_pos = 0.0
    i = 0
    while i < n:
        j = i + 1
        while j < n and scores[order[j]] == scores[order[i]]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for pos in range(i, j):
            if labels[order[pos]]:
                rank_sum_pos += avg_rank
        i = j
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def read_raw_titles(data_dir, dataset):
    encoding = "latin1" if dataset == "CiteULike" else "utf-8"
    raw = pd.read_csv(Path(data_dir) / "raw-data.csv", encoding=encoding)
    title_col = "title" if "title" in raw.columns else next((c for c in raw.columns if "title" in c.lower()), None)
    if title_col is None:
        return [f"item {idx}" for idx in range(len(raw))]
    return raw[title_col].fillna("").astype(str).tolist()


def load_counts(data_dir):
    with (Path(data_dir) / "n_user_item.pkl").open("rb") as f:
        info = pickle.load(f)
    return int(info["user"]), int(info["item"])


def read_edges(path):
    if not Path(path).exists():
        return pd.DataFrame(columns=["user", "item"])
    df = pd.read_csv(path)
    if not {"user", "item"}.issubset(df.columns):
        return pd.DataFrame(columns=["user", "item"])
    return df[["user", "item"]].astype(int)


def all_known_pairs(data_dir):
    known = set()
    for path in Path(data_dir).glob("*.csv"):
        try:
            df = pd.read_csv(path, usecols=["user", "item"])
        except Exception:
            continue
        known.update((int(row.user), int(row.item)) for row in df.itertuples(index=False))
    return known


def domain_words(dataset):
    if dataset == "ml-1m":
        return "movie", "movies"
    return "paper", "papers"


def build_examples(args):
    rng = random.Random(args.seed)
    data_dir = Path(args.data_dir)
    n_users, n_items = load_counts(data_dir)
    titles = read_raw_titles(data_dir, args.dataset)
    known = all_known_pairs(data_dir)

    positives = []
    for split_name, file_name in [("strict_cold", "cold_item_test.csv"), ("warmup", "warmup_test.csv")]:
        df = read_edges(data_dir / file_name)
        if args.max_per_split and len(df) > args.max_per_split:
            df = df.sample(n=args.max_per_split, random_state=args.seed)
        for row in df.itertuples(index=False):
            positives.append((int(row.user), int(row.item), split_name))

    examples = []
    pairs = []
    singular, plural = domain_words(args.dataset)
    instruction = (
        f"No user interaction history is provided. Estimate the probability that the specified user "
        f"will like the target {singular}. Return only a numeric probability in [0, 1]."
    )

    for user, item, split_name in positives:
        title = titles[item] if 0 <= item < len(titles) else f"item {item}"
        examples.append(
            {
                "instruction": instruction,
                "input": f'User id: "{user}". Target {singular}: "{title}".',
                "output": "1",
                "user_id": user,
                "item_id": item,
                "entity_type": split_name,
                "label": 1,
            }
        )
        pairs.append({"user": user, "item": item, "entity_type": split_name, "label": 1, "title": title})

        neg_user = None
        for _ in range(500):
            candidate = rng.randrange(n_users)
            if (candidate, item) not in known:
                neg_user = candidate
                break
        if neg_user is None:
            continue
        examples.append(
            {
                "instruction": instruction,
                "input": f'User id: "{neg_user}". Target {singular}: "{title}".',
                "output": "0",
                "user_id": neg_user,
                "item_id": item,
                "entity_type": f"{split_name}_negative",
                "label": 0,
            }
        )
        pairs.append({"user": neg_user, "item": item, "entity_type": f"{split_name}_negative", "label": 0, "title": title})

    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(examples, indent=2, ensure_ascii=False), encoding="utf-8")
    pd.DataFrame(pairs).to_csv(args.output_pairs_csv, index=False)
    print(f"Saved {len(examples)} no-context probe examples to {args.output_json}")


def evaluate_predictions(args):
    examples = json.loads(Path(args.output_json).read_text(encoding="utf-8"))
    predictions = []
    with Path(args.prediction_jsonl).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    predictions.append(parse_prob(json.loads(line).get("predict")))
                except json.JSONDecodeError:
                    predictions.append(None)
    usable = min(len(examples), len(predictions))
    labels = [int(examples[idx]["label"]) for idx in range(usable)]
    scores = predictions[:usable]
    pos_scores = [s for s, l in zip(scores, labels) if s is not None and l == 1]
    neg_scores = [s for s, l in zip(scores, labels) if s is not None and l == 0]
    summary = {
        "examples": len(examples),
        "predictions": len(predictions),
        "aligned": usable,
        "valid_predictions": sum(s is not None for s in scores),
        "positive_mean": sum(pos_scores) / len(pos_scores) if pos_scores else None,
        "negative_mean": sum(neg_scores) / len(neg_scores) if neg_scores else None,
        "auc": auc_from_labels(scores, labels),
    }
    Path(args.summary_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary_json).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Build/evaluate no-context memorization probes.")
    parser.add_argument("--dataset", required=True, choices=["CiteULike", "ml-1m"])
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_pairs_csv", required=True)
    parser.add_argument("--prediction_jsonl", default=None)
    parser.add_argument("--summary_json", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_per_split", type=int, default=300)
    args = parser.parse_args()

    build_examples(args)
    if args.prediction_jsonl:
        if not args.summary_json:
            raise SystemExit("--summary_json is required when --prediction_jsonl is passed")
        evaluate_predictions(args)


if __name__ == "__main__":
    main()
