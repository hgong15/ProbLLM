#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pickle
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
        description="Build clean TextEmb-only candidates with frozen LLaMA text embeddings."
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--dataset", default="CiteULike")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--model_name_or_path", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--top_m", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--torch_dtype", choices=["float16", "bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--pooling", choices=["mean", "last"], default="mean")
    parser.add_argument("--item_text", choices=["title", "title_abstract"], default="title")
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


def load_target_item_types(data_dir: Path) -> dict[int, str]:
    target: dict[int, str] = {}
    for name in ("cold_item_val.csv", "cold_item_test.csv"):
        for item in read_pairs(data_dir / name)["item"].unique().tolist():
            target[int(item)] = "strict_cold"
    for name in ("warmup_val.csv", "warmup_test.csv"):
        for item in read_pairs(data_dir / name)["item"].unique().tolist():
            target[int(item)] = "warmup"
    return target


def train_users_by_item(train_df: pd.DataFrame) -> dict[int, set[int]]:
    grouped: dict[int, set[int]] = defaultdict(set)
    for user, item in train_df[["user", "item"]].itertuples(index=False):
        grouped[int(item)].add(int(user))
    return grouped


def citeulike_item_texts(data_dir: Path, mode: str) -> list[str]:
    raw = pd.read_csv(data_dir / "raw-data.csv", encoding="latin1")
    titles = raw["title"].fillna("").astype(str).tolist()
    if mode == "title":
        return titles
    abstracts = raw.get("raw.abstract", pd.Series([""] * len(raw))).fillna("").astype(str).tolist()
    return [(title + ". " + abstract).strip() for title, abstract in zip(titles, abstracts)]


def encode_texts(args: argparse.Namespace, texts: list[str], save_path: Path) -> torch.Tensor:
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


def build_topm(user_emb: torch.Tensor, item_emb: torch.Tensor, target_item_types: dict[int, str], excluded: dict[int, set[int]], top_m: int) -> pd.DataFrame:
    user_emb = F.normalize(user_emb.float(), p=2, dim=1)
    item_emb = F.normalize(item_emb.float(), p=2, dim=1)
    scores = item_emb @ user_emb.t()

    rows = []
    for item in tqdm(sorted(target_item_types), desc="Selecting clean text candidates"):
        item_scores = scores[item].clone()
        blocked = excluded.get(int(item), set())
        if blocked:
            item_scores[torch.as_tensor(sorted(blocked), dtype=torch.long)] = -torch.inf
        take = min(max(top_m + len(blocked), top_m), item_scores.numel())
        ranked = torch.topk(item_scores, k=take).indices.tolist()
        selected = []
        for user in ranked:
            if int(user) in blocked:
                continue
            selected.append(int(user))
            if len(selected) >= top_m:
                break
        for user in selected:
            rows.append({"user": user, "item": int(item), "entity_type": target_item_types[int(item)]})
    return pd.DataFrame(rows, columns=["user", "item", "entity_type"])


def main() -> None:
    args = parse_args()
    data_dir = args.root / "data" / args.dataset
    args.output_dir.mkdir(parents=True, exist_ok=True)

    item_emb_path = args.output_dir / "frozen_llama_item_text_emb.pt"
    user_emb_path = args.output_dir / "frozen_llama_user_profile_emb.pt"
    top20_path = args.output_dir / "top20.csv"
    pseudo_path = args.output_dir / "predicted_cold_item_interaction.csv"
    meta_path = args.output_dir / "clean_textemb_meta.json"

    item_texts = citeulike_item_texts(data_dir, args.item_text)
    with (data_dir / "train_user_preference_list.pkl").open("rb") as handle:
        user_texts = pickle.load(handle)

    item_emb = encode_texts(args, item_texts, item_emb_path)
    user_emb = encode_texts(args, user_texts, user_emb_path)

    target_item_types = load_target_item_types(data_dir)
    train_df = read_pairs(data_dir / "warm_emb.csv")
    excluded = train_users_by_item(train_df)
    top20 = build_topm(user_emb, item_emb, target_item_types, excluded, args.top_m)
    top20.to_csv(top20_path, index=False)
    pseudo = top20.copy()
    pseudo["probability"] = 1.0
    pseudo.to_csv(pseudo_path, index=False)

    meta = {
        "method": "TextEmb-only-7B-clean",
        "definition": "Frozen base LLaMA text embeddings + cosine top-M candidates; no LoRA, no pairwise LLM scoring, no LlamaHead filtering.",
        "dataset": args.dataset,
        "seed": args.seed,
        "model_name_or_path": str(args.model_name_or_path),
        "top_m": args.top_m,
        "pooling": args.pooling,
        "item_text": args.item_text,
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "rows": int(len(top20)),
        "users": int(top20["user"].nunique()),
        "items": int(top20["item"].nunique()),
        "strict_rows": int((top20["entity_type"] == "strict_cold").sum()),
        "warmup_rows": int((top20["entity_type"] == "warmup").sum()),
        "item_embedding": str(item_emb_path),
        "user_embedding": str(user_emb_path),
        "top20_csv": str(top20_path),
        "pseudo_csv": str(pseudo_path),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2), flush=True)


if __name__ == "__main__":
    main()
