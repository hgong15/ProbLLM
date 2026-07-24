#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build text-neighbor top-M user candidates for LLM augmentation."
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--dataset", default="CiteULike")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--model_name_or_path", type=Path, required=True)
    parser.add_argument("--embedding_cache_dir", type=Path, default=None)
    parser.add_argument("--output_csv", type=Path, required=True)
    parser.add_argument("--meta_json", type=Path, default=None)
    parser.add_argument("--top_m", type=int, default=20)
    parser.add_argument("--neighbor_k", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--torch_dtype", choices=["float16", "bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--pooling", choices=["mean", "last"], default="mean")
    parser.add_argument("--item_text", choices=["title", "title_abstract"], default="title")
    parser.add_argument("--score_agg", choices=["sum", "max"], default="sum")
    parser.add_argument("--force_reencode", action="store_true")
    return parser.parse_args()


def resolve_dtype(name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def read_pairs(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["user", "item"])
    df = pd.read_csv(path)
    if df.empty:
        return pd.DataFrame(columns=["user", "item"])
    return df[["user", "item"]].astype({"user": int, "item": int})


def citeulike_item_texts(data_dir: Path, mode: str) -> list[str]:
    raw = pd.read_csv(data_dir / "raw-data.csv", encoding="latin1")
    titles = raw["title"].fillna("").astype(str).tolist()
    if mode == "title":
        return titles
    abstracts = raw.get("raw.abstract", pd.Series([""] * len(raw))).fillna("").astype(str).tolist()
    return [(title + ". " + abstract).strip() for title, abstract in zip(titles, abstracts)]


def encode_item_texts(args: argparse.Namespace, texts: list[str], save_path: Path) -> torch.Tensor:
    if save_path.exists() and not args.force_reencode:
        return torch.load(save_path, map_location="cpu").float()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=resolve_dtype(args.torch_dtype),
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    device = next(model.parameters()).device

    chunks = []
    with torch.inference_mode():
        for start in tqdm(range(0, len(texts), args.batch_size), desc=f"Encoding {save_path.name}"):
            batch = [text if str(text).strip() else " " for text in texts[start : start + args.batch_size]]
            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=args.max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            outputs = model(**encoded, output_hidden_states=True)
            hidden = outputs.hidden_states[-1].float()
            mask = encoded["attention_mask"].float()
            if args.pooling == "mean":
                pooled = (hidden * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1.0)
            else:
                positions = torch.arange(mask.shape[1], device=mask.device).unsqueeze(0)
                last_nonpad = (mask.long() * positions).max(dim=1).values
                pooled = hidden[torch.arange(hidden.shape[0], device=hidden.device), last_nonpad]
            chunks.append(pooled.detach().cpu())
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    emb = torch.cat(chunks, dim=0).float()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(emb, save_path)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return emb


def target_item_types(data_dir: Path) -> dict[int, str]:
    target: dict[int, str] = {}
    for name in ("cold_item_val.csv", "cold_item_test.csv", "strict_cold_item_val.csv", "strict_cold_item_test.csv"):
        for item in read_pairs(data_dir / name)["item"].unique().tolist():
            target[int(item)] = "strict_cold"
    for name in ("warmup_val.csv", "warmup_test.csv"):
        for item in read_pairs(data_dir / name)["item"].unique().tolist():
            target[int(item)] = "warmup"
    return target


def build_train_indexes(train_df: pd.DataFrame):
    users_by_item: dict[int, list[int]] = defaultdict(list)
    items_by_user: dict[int, set[int]] = defaultdict(set)
    for user, item in train_df[["user", "item"]].itertuples(index=False):
        user_id = int(user)
        item_id = int(item)
        users_by_item[item_id].append(user_id)
        items_by_user[user_id].add(item_id)
    return users_by_item, items_by_user


def choose_users_for_item(
    item: int,
    item_emb: torch.Tensor,
    warm_items: torch.Tensor,
    warm_emb: torch.Tensor,
    users_by_item: dict[int, list[int]],
    items_by_user: dict[int, set[int]],
    neighbor_k: int,
    top_m: int,
    score_agg: str,
) -> tuple[list[tuple[int, float]], list[int]]:
    sims = (item_emb[item : item + 1] @ warm_emb.t()).squeeze(0)
    same_item = (warm_items == item).nonzero(as_tuple=False).flatten()
    if same_item.numel():
        sims[same_item] = -torch.inf

    take = min(neighbor_k, int(warm_items.numel()))
    values, local_indices = torch.topk(sims, k=take)
    scores: dict[int, float] = defaultdict(float)
    neighbors: list[int] = []
    for sim, local in zip(values.tolist(), local_indices.tolist()):
        if not np.isfinite(sim):
            continue
        warm_item = int(warm_items[local])
        neighbors.append(warm_item)
        for user in users_by_item.get(warm_item, []):
            if item in items_by_user.get(user, set()):
                continue
            contribution = max(float(sim), 0.0)
            if score_agg == "max":
                scores[user] = max(scores[user], contribution)
            else:
                scores[user] += contribution

    ranked = sorted(scores.items(), key=lambda row: (-row[1], row[0]))[:top_m]
    return ranked, neighbors


def build_candidates(args: argparse.Namespace, item_emb: torch.Tensor, data_dir: Path) -> tuple[pd.DataFrame, dict]:
    train_df = read_pairs(data_dir / "warm_emb.csv")
    users_by_item, items_by_user = build_train_indexes(train_df)
    warm_items = torch.as_tensor(sorted(users_by_item.keys()), dtype=torch.long)
    if warm_items.numel() == 0:
        raise ValueError("No warm training items found in warm_emb.csv")

    item_emb = F.normalize(item_emb.float(), p=2, dim=1)
    warm_emb = item_emb[warm_items]
    targets = target_item_types(data_dir)

    rows = []
    short_items = []
    neighbor_examples = {}
    for item in tqdm(sorted(targets), desc="Selecting text-neighbor candidates"):
        ranked, neighbors = choose_users_for_item(
            int(item),
            item_emb,
            warm_items,
            warm_emb,
            users_by_item,
            items_by_user,
            args.neighbor_k,
            args.top_m,
            args.score_agg,
        )
        if len(neighbor_examples) < 5:
            neighbor_examples[str(item)] = neighbors[:10]
        if len(ranked) < args.top_m:
            short_items.append({"item": int(item), "entity_type": targets[int(item)], "candidates": len(ranked)})
        for user, score in ranked:
            rows.append(
                {
                    "user": int(user),
                    "item": int(item),
                    "entity_type": targets[int(item)],
                    "candidate_score": float(score),
                }
            )

    df = pd.DataFrame(rows, columns=["user", "item", "entity_type", "candidate_score"])
    meta = {
        "method": "Text-neighbor top-M",
        "definition": (
            "For each strict-cold/warm-up item, find nearest warm training items by frozen "
            "LLaMA item-text cosine similarity, aggregate those warm items' training users, "
            "and select top-M users. No LoRA or pairwise LLM is used for candidate generation."
        ),
        "dataset": args.dataset,
        "seed": args.seed,
        "top_m": args.top_m,
        "neighbor_k": args.neighbor_k,
        "score_agg": args.score_agg,
        "train_interactions": int(len(train_df)),
        "warm_source_items": int(warm_items.numel()),
        "target_items": int(len(targets)),
        "strict_rows": int((df["entity_type"] == "strict_cold").sum()) if not df.empty else 0,
        "warmup_rows": int((df["entity_type"] == "warmup").sum()) if not df.empty else 0,
        "short_items": short_items[:20],
        "short_item_count": len(short_items),
        "neighbor_examples": neighbor_examples,
    }
    return df, meta


def main() -> None:
    args = parse_args()
    data_dir = args.root / "data" / args.dataset
    cache_dir = args.embedding_cache_dir or args.output_csv.parent
    item_emb_path = cache_dir / "frozen_llama_item_text_emb.pt"

    item_texts = citeulike_item_texts(data_dir, args.item_text)
    item_emb = encode_item_texts(args, item_texts, item_emb_path)
    topm, meta = build_candidates(args, item_emb, data_dir)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    topm.to_csv(args.output_csv, index=False)
    meta.update(
        {
            "model_name_or_path": str(args.model_name_or_path),
            "item_embedding": str(item_emb_path),
            "item_text": args.item_text,
            "pooling": args.pooling,
            "max_length": args.max_length,
            "batch_size": args.batch_size,
            "rows": int(len(topm)),
            "users": int(topm["user"].nunique()) if not topm.empty else 0,
            "items": int(topm["item"].nunique()) if not topm.empty else 0,
            "output_csv": str(args.output_csv),
        }
    )
    meta_path = args.meta_json or args.output_csv.with_suffix(".meta.json")
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2), flush=True)


if __name__ == "__main__":
    main()
