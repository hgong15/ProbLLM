import argparse
import json
import os
import pickle
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm


class LlamaHead(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.user_mlp = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )
        self.item_mlp = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

    def calculate_score(self, user_origin_emb, item_origin_emb):
        user_content_emb = self.user_mlp(user_origin_emb)
        item_content_emb = self.item_mlp(item_origin_emb)
        return torch.matmul(user_content_emb, item_content_emb.t())


def read_pairs(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame(columns=["user", "item"])


def load_user_targets(data_dir: Path) -> dict[int, str]:
    targets: dict[int, str] = {}
    for filename in ("cold_user_val.csv", "cold_user_test.csv"):
        df = read_pairs(data_dir / filename)
        for user in sorted(df["user"].unique().tolist()) if len(df) else []:
            targets[int(user)] = "strict_cold"
    for filename in ("warmup_val.csv", "warmup_test.csv"):
        df = read_pairs(data_dir / filename)
        for user in sorted(df["user"].unique().tolist()) if len(df) else []:
            targets[int(user)] = "warmup"
    return targets


def main():
    parser = argparse.ArgumentParser(description="Generate cold-user item candidates using LLM/ALDI/hybrid scores.")
    parser.add_argument("--dataset", default="ml-1m")
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--filter_source", choices=["llm", "aldi", "hybrid"], default="hybrid")
    parser.add_argument("--normalize_filter", action="store_true")
    parser.add_argument("--hidden_size", type=int, default=1024)
    parser.add_argument("--output_size", type=int, default=200)
    parser.add_argument("--output_csv", default=None)
    parser.add_argument("--output_json", default=None)
    parser.add_argument("--diagnostics_json", default=None)
    args = parser.parse_args()

    data_dir = Path(args.data_dir or Path("data") / args.dataset)
    output_csv = Path(args.output_csv) if args.output_csv else data_dir / "top20.csv"
    output_json = Path(args.output_json) if args.output_json else data_dir / f"{args.dataset}_eval.json"
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    with (data_dir / "convert_dict.pkl").open("rb") as f:
        para = pickle.load(f)
    if para.get("cold_object") != "user":
        raise ValueError(f"Expected cold_object=user, got {para.get('cold_object')!r}")

    targets = load_user_targets(data_dir)
    target_users = sorted(targets)
    if not target_users:
        raise ValueError("No cold/warmup target users found.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    user_origin_emb = torch.load(data_dir / "llm_user_content_emb.pt", map_location="cpu").float().to(device)
    item_origin_emb = torch.load(data_dir / "llm_item_content_emb.pt", map_location="cpu").float().to(device)
    input_size = user_origin_emb.shape[1]
    if input_size != item_origin_emb.shape[1]:
        raise ValueError(f"User/item LLM embedding dims differ: {user_origin_emb.shape} vs {item_origin_emb.shape}")

    llama_model = LlamaHead(input_size, args.hidden_size, args.output_size).to(device)
    llama_model.load_state_dict(torch.load(data_dir / "llama_head.bin", map_location="cpu"))
    llama_model.eval()
    with torch.no_grad():
        if args.normalize_filter:
            mapped_user = F.normalize(llama_model.user_mlp(user_origin_emb), p=2, dim=1)
            mapped_item = F.normalize(llama_model.item_mlp(item_origin_emb), p=2, dim=1)
            llm_score = mapped_user @ mapped_item.t()
        else:
            llm_score = llama_model.calculate_score(user_origin_emb, item_origin_emb)

    if args.filter_source == "llm":
        all_score = llm_score
    else:
        aldi_user = torch.load(data_dir / "ALDI_user_emb.pt", map_location="cpu").float().to(device)
        aldi_item = torch.load(data_dir / "ALDI_item_emb.pt", map_location="cpu").float().to(device)
        aldi_score = aldi_user @ aldi_item.t()
        if args.filter_source == "aldi":
            all_score = aldi_score
        else:
            all_score = llm_score + aldi_score

    train_df = read_pairs(data_dir / "warm_emb.csv")
    visible_items = sorted(train_df["item"].unique().tolist())
    visible_items_tensor = torch.as_tensor(visible_items, dtype=torch.long, device=device)
    observed_by_user = train_df.groupby("user")["item"].agg(set).to_dict() if len(train_df) else {}

    if args.dataset == "CiteULike":
        item_content = pd.read_csv(data_dir / "raw-data.csv", encoding="latin1")
        instruction = (
            "Given the user's interaction paper set, predict the probability "
            "(a value between 0 and 1, e.g., 0.85) that the user will like the target paper."
        )
        target_name = "target paper"
    else:
        item_content = pd.read_csv(data_dir / "raw-data.csv")
        instruction = (
            "Given the user's interaction movie set, predict the probability "
            "(a value between 0 and 1, e.g., 0.32) that the user will like the target movie."
        )
        target_name = "target movie"

    with (data_dir / "train_user_preference_list.pkl").open("rb") as f:
        train_user_preference_list = pickle.load(f)

    rows = []
    examples = []
    for user in tqdm(target_users, desc="Selecting cold-user item candidates"):
        blocked = observed_by_user.get(int(user), set())
        scores = all_score[int(user), visible_items_tensor]
        candidate_pool_size = min(
            len(visible_items),
            max(args.topk * 5, args.topk + len(blocked) + 20),
        )
        _, ranked_local = torch.topk(scores, candidate_pool_size)
        selected = []
        for local_idx in ranked_local.tolist():
            item = int(visible_items[local_idx])
            if item in blocked:
                continue
            selected.append(item)
            if len(selected) >= args.topk:
                break
        entity_type = targets[int(user)]
        for item in selected:
            rows.append({"user": int(user), "item": int(item), "entity_type": entity_type})
            examples.append(
                {
                    "instruction": instruction,
                    "input": (
                        f'User preference: "{train_user_preference_list[int(user)]}", '
                        f'What is the probability the user will like the {target_name} '
                        f'"{item_content.iloc[int(item)].title}"?'
                    ),
                    "output": "1",
                    "user_id": int(user),
                    "item_id": int(item),
                    "entity_type": entity_type,
                }
            )

    out = pd.DataFrame(rows, columns=["user", "item", "entity_type"])
    out.to_csv(output_csv, index=False)
    output_json.write_text(json.dumps(examples, indent=4, ensure_ascii=False), encoding="utf-8")

    diagnostics = {
        "dataset": args.dataset,
        "cold_object": "user",
        "filter_source": args.filter_source,
        "normalize_filter": args.normalize_filter,
        "topk": args.topk,
        "target_users": len(target_users),
        "rows": len(out),
        "unique_users": int(out["user"].nunique()) if len(out) else 0,
        "unique_items": int(out["item"].nunique()) if len(out) else 0,
        "entity_type_counts": out["entity_type"].value_counts().to_dict() if len(out) else {},
        "output_csv": str(output_csv),
        "output_json": str(output_json),
    }
    print(json.dumps(diagnostics, indent=2, ensure_ascii=False))
    if args.diagnostics_json:
        Path(args.diagnostics_json).write_text(json.dumps(diagnostics, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
