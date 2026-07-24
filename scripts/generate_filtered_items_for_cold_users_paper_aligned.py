#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm


class LlamaHead(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
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

    def score_matrix(self, user_emb: torch.Tensor, item_emb: torch.Tensor, normalize: bool = False):
        users = self.user_mlp(user_emb)
        items = self.item_mlp(item_emb)
        if normalize:
            users = F.normalize(users, p=2, dim=1)
            items = F.normalize(items, p=2, dim=1)
        return users @ items.t()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate paper-aligned MovieLens cold-user hybrid candidates.")
    parser.add_argument("--data_dir", type=Path, default=Path("data/ml-1m"))
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--filter_source", choices=["llm", "aldi", "hybrid"], default="hybrid")
    parser.add_argument("--normalize_llm", action="store_true")
    parser.add_argument("--normalize_components", action="store_true")
    parser.add_argument("--hidden_size", type=int, default=1024)
    parser.add_argument("--output_size", type=int, default=200)
    parser.add_argument("--llama_head", default="llama_head_user_cold.bin")
    parser.add_argument("--llm_user_emb", default="llm_user_side_emb.pt")
    parser.add_argument("--llm_item_emb", default="llm_item_side_emb.pt")
    parser.add_argument("--aldi_user_emb", default="user_side_aldi_user_emb.pt")
    parser.add_argument("--aldi_item_emb", default="user_side_aldi_item_emb.pt")
    parser.add_argument("--user_text_file", default="user_side_context_list.pkl")
    parser.add_argument("--output_csv", type=Path, default=None)
    parser.add_argument("--output_json", type=Path, default=None)
    parser.add_argument("--diagnostics_json", type=Path, default=None)
    return parser.parse_args()


def read_pairs(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["user", "item"])
    df = pd.read_csv(path)
    if df.empty:
        return pd.DataFrame(columns=["user", "item"])
    return df[["user", "item"]].astype({"user": int, "item": int})


def target_user_types(data_dir: Path) -> dict[int, str]:
    targets: dict[int, str] = {}
    for name in ("cold_user_val.csv", "cold_user_test.csv"):
        for user in read_pairs(data_dir / name)["user"].unique().tolist():
            targets[int(user)] = "strict_cold"
    for name in ("warmup_val.csv", "warmup_test.csv"):
        for user in read_pairs(data_dir / name)["user"].unique().tolist():
            targets[int(user)] = "warmup"
    return targets


def component_zscore(score: torch.Tensor) -> torch.Tensor:
    mean = score.mean(dim=1, keepdim=True)
    std = score.std(dim=1, keepdim=True).clamp_min(1e-6)
    return (score - mean) / std


def candidate_hit_stats(top20: pd.DataFrame, data_dir: Path) -> dict:
    stats = {}
    gt_files = {
        "strict_cold_test": data_dir / "cold_user_test.csv",
        "warmup_test": data_dir / "warmup_test.csv",
        "strict_cold_val": data_dir / "cold_user_val.csv",
        "warmup_val": data_dir / "warmup_val.csv",
    }
    cand_by_user = top20.groupby("user")["item"].agg(set).to_dict() if len(top20) else {}
    for name, path in gt_files.items():
        gt = read_pairs(path)
        if gt.empty:
            stats[name] = {"users": 0, "gt_pairs": 0, "hits": 0, "recall": 0.0, "user_hit_rate": 0.0}
            continue
        hits = 0
        user_hit = set()
        gt_user = set()
        for user, item in gt[["user", "item"]].itertuples(index=False):
            user = int(user)
            item = int(item)
            gt_user.add(user)
            if item in cand_by_user.get(user, set()):
                hits += 1
                user_hit.add(user)
        stats[name] = {
            "users": len(gt_user),
            "gt_pairs": int(len(gt)),
            "hits": int(hits),
            "recall": float(hits / len(gt)) if len(gt) else 0.0,
            "user_hit_rate": float(len(user_hit) / len(gt_user)) if gt_user else 0.0,
        }
    return stats


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir
    output_csv = args.output_csv or data_dir / "top20_user_cold_paper_aligned.csv"
    output_json = args.output_json or data_dir / "ml-1m_user_eval.json"
    diagnostics_json = args.diagnostics_json or data_dir / "user_cold_candidate_diagnostics.json"
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    targets = target_user_types(data_dir)
    if not targets:
        raise ValueError("No strict-cold or warm-up users found.")
    target_users = sorted(targets)
    train_df = read_pairs(data_dir / "warm_emb.csv")
    visible_items = sorted(train_df["item"].unique().tolist())
    visible_tensor = torch.as_tensor(visible_items, dtype=torch.long)
    observed_by_user = train_df.groupby("user")["item"].agg(set).to_dict() if len(train_df) else {}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    llm_user = torch.load(data_dir / args.llm_user_emb, map_location="cpu").float().to(device)
    llm_item = torch.load(data_dir / args.llm_item_emb, map_location="cpu").float().to(device)
    head = LlamaHead(llm_user.shape[1], args.hidden_size, args.output_size).to(device)
    head.load_state_dict(torch.load(data_dir / args.llama_head, map_location="cpu"))
    head.eval()
    with torch.no_grad():
        llm_score_all = head.score_matrix(llm_user, llm_item, normalize=args.normalize_llm)
        llm_score = llm_score_all[torch.as_tensor(target_users, dtype=torch.long, device=device)][:, visible_tensor.to(device)]

    if args.filter_source == "llm":
        score = llm_score
        aldi_score = None
    else:
        aldi_user = torch.load(data_dir / args.aldi_user_emb, map_location="cpu").float().to(device)
        aldi_item = torch.load(data_dir / args.aldi_item_emb, map_location="cpu").float().to(device)
        aldi_score = (aldi_user[torch.as_tensor(target_users, dtype=torch.long, device=device)] @ aldi_item[visible_tensor.to(device)].t())
        if args.filter_source == "aldi":
            score = aldi_score
        else:
            if args.normalize_components:
                score = component_zscore(llm_score) + component_zscore(aldi_score)
            else:
                score = llm_score + aldi_score

    item_content = pd.read_csv(data_dir / "raw-data.csv")
    with (data_dir / args.user_text_file).open("rb") as handle:
        user_context = pickle.load(handle)
    instruction = (
        "Given the user's profile and observed movie interactions, predict the probability "
        "(a value between 0 and 1, e.g., 0.32) that the user will like the target movie."
    )

    rows = []
    examples = []
    for row_idx, user in enumerate(tqdm(target_users, desc="Selecting cold-user candidates")):
        blocked = observed_by_user.get(int(user), set())
        ranked = torch.argsort(score[row_idx], descending=True).detach().cpu().tolist()
        selected = []
        for local_idx in ranked:
            item = int(visible_items[local_idx])
            if item in blocked:
                continue
            selected.append(item)
            if len(selected) >= args.topk:
                break
        for item in selected:
            rows.append({"user": int(user), "item": int(item), "entity_type": targets[int(user)]})
            examples.append(
                {
                    "instruction": instruction,
                    "input": f'{user_context[int(user)]}, target movie: "{item_content.iloc[int(item)].title}".',
                    "output": "1",
                    "user_id": int(user),
                    "item_id": int(item),
                    "entity_type": targets[int(user)],
                }
            )

    out = pd.DataFrame(rows, columns=["user", "item", "entity_type"])
    out.to_csv(output_csv, index=False)
    output_json.write_text(json.dumps(examples, indent=4, ensure_ascii=False), encoding="utf-8")

    diagnostics = {
        "dataset": "ml-1m",
        "cold_object": "user",
        "filter_source": args.filter_source,
        "normalize_llm": bool(args.normalize_llm),
        "normalize_components": bool(args.normalize_components),
        "topk": args.topk,
        "target_users": len(target_users),
        "rows": int(len(out)),
        "unique_users": int(out["user"].nunique()) if len(out) else 0,
        "unique_items": int(out["item"].nunique()) if len(out) else 0,
        "entity_type_counts": out["entity_type"].value_counts().to_dict() if len(out) else {},
        "candidate_hit": candidate_hit_stats(out, data_dir),
        "visible_items": len(visible_items),
        "output_csv": str(output_csv),
        "output_json": str(output_json),
    }
    diagnostics_json.write_text(json.dumps(diagnostics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(diagnostics, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
