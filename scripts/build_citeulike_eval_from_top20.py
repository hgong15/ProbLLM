import argparse
import json
import pickle
from pathlib import Path

import pandas as pd


def title_lookup(item_content: pd.DataFrame):
    if "item" in item_content.columns:
        return item_content.set_index("item")["title"].astype(str).to_dict()
    return item_content["title"].astype(str).to_dict()


def load_user_preferences(data_dir: Path, item_content: pd.DataFrame, max_user_id: int = -1, max_history: int = 20):
    path = data_dir / "train_user_preference_list.pkl"
    if path.exists():
        with path.open("rb") as f:
            return pickle.load(f)

    train_path = data_dir / "warm_emb.csv"
    if not train_path.exists():
        raise FileNotFoundError(f"Missing {path} and fallback train file {train_path}")
    train = pd.read_csv(train_path)
    if not {"user", "item"}.issubset(train.columns):
        raise ValueError(f"{train_path} must contain user,item columns")

    titles = title_lookup(item_content)
    max_user = max(int(train["user"].max()) if len(train) else -1, int(max_user_id))
    preferences = [""] * (max_user + 1)
    for user, group in train.groupby("user", sort=False):
        seen_titles = []
        for item in group["item"].astype(int).tolist()[:max_history]:
            title = titles.get(item, "")
            if title and title != "nan":
                seen_titles.append(title)
        preferences[int(user)] = "; ".join(seen_titles) if seen_titles else "No observed historical items"
    return preferences


def main():
    parser = argparse.ArgumentParser(description="Build LLM eval JSON from a saved top20.csv.")
    parser.add_argument("--data_dir", default="data/CiteULike")
    parser.add_argument("--top20_csv", required=True)
    parser.add_argument("--output_json", default=None)
    parser.add_argument("--raw_data_csv", default=None)
    parser.add_argument("--domain", choices=["paper", "movie", "book", "product"], default="paper")
    parser.add_argument(
        "--prompt_variant",
        choices=["default", "concise", "numeric_only", "preference_match"],
        default="default",
        help="Prompt wording variant for prompt-sensitivity diagnostics.",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    top20_path = Path(args.top20_csv)
    output_path = Path(args.output_json) if args.output_json else data_dir / "CiteULike_eval.json"

    top20_df = pd.read_csv(top20_path)
    raw_data_csv = Path(args.raw_data_csv) if args.raw_data_csv else data_dir / "raw-data.csv"
    if args.domain == "paper":
        item_content = pd.read_csv(raw_data_csv, encoding="latin1")
    else:
        item_content = pd.read_csv(raw_data_csv)
    max_user_id = int(top20_df["user"].max()) if len(top20_df) else -1
    train_user_preference_list = load_user_preferences(data_dir, item_content, max_user_id=max_user_id)
    titles = title_lookup(item_content)

    domain_words = {
        "paper": ("paper", "papers", "0.85"),
        "movie": ("movie", "movies", "0.32"),
        "book": ("book", "books", "0.32"),
        "product": ("product", "products", "0.32"),
    }
    singular, plural, example_prob = domain_words[args.domain]
    target_text = f"target {singular}"
    if args.prompt_variant == "default":
        instruction = (
            f"Given the user's interaction {singular} set, predict the probability "
            f"(a value between 0 and 1, e.g., {example_prob}) that the user will like the target {singular}."
        )
    elif args.prompt_variant == "concise":
        instruction = (
            f"Estimate how likely the user is to like the target {singular}. "
            "Return a probability between 0 and 1."
        )
    elif args.prompt_variant == "numeric_only":
        instruction = (
            f"Given the user's past {plural}, score the target {singular}. "
            "Output only one numeric probability in [0, 1]."
        )
    else:
        instruction = (
            f"Compare the target {singular} with the user's historical {plural}. "
            "Return the probability that the target matches the user's preference."
        )
    examples = []
    for row in top20_df.itertuples(index=False):
        user_id = int(row.user)
        item_id = int(row.item)
        entity_type = getattr(row, "entity_type", "")
        examples.append(
            {
                "instruction": instruction,
                "input": (
                    f'User preference: "{train_user_preference_list[user_id]}", '
                    f'What is the probability the user will like the {target_text} '
                    f'"{titles.get(item_id, item_content.iloc[item_id].title)}"?'
                ),
                "output": "1",
                "user_id": user_id,
                "item_id": item_id,
                "entity_type": entity_type,
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(examples, indent=4, ensure_ascii=False), encoding="utf-8")
    print(f"Saved {len(examples)} examples to {output_path}")


if __name__ == "__main__":
    main()
