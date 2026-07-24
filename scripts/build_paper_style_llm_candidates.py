#!/usr/bin/env python
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


def read_pairs(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["user", "item"])
    return pd.read_csv(path, usecols=["user", "item"]).astype({"user": int, "item": int})


def load_targets(data_dir: Path, cold_object: str) -> dict[int, str]:
    target_col = "item" if cold_object == "item" else "user"
    strict_prefix = "cold_item" if cold_object == "item" else "cold_user"
    targets: dict[int, str] = {}
    for split in ("val", "test"):
        strict = read_pairs(data_dir / f"{strict_prefix}_{split}.csv")
        for entity in strict[target_col].unique().tolist() if len(strict) else []:
            targets[int(entity)] = "strict_cold"
    for split in ("val", "test"):
        warmup = read_pairs(data_dir / f"warmup_{split}.csv")
        for entity in warmup[target_col].unique().tolist() if len(warmup) else []:
            targets[int(entity)] = "warmup"
    return dict(sorted(targets.items()))


def load_observed(data_dir: Path, cold_object: str) -> dict[int, set[int]]:
    train = read_pairs(data_dir / "warm_emb.csv")
    if len(train) == 0:
        return {}
    if cold_object == "item":
        return {int(k): set(map(int, v)) for k, v in train.groupby("item")["user"].agg(set).to_dict().items()}
    return {int(k): set(map(int, v)) for k, v in train.groupby("user")["item"].agg(set).to_dict().items()}


def load_convert(data_dir: Path) -> dict:
    path = data_dir / "convert_dict.pkl"
    if not path.exists():
        return {}
    with path.open("rb") as f:
        obj = pickle.load(f)
    return obj if isinstance(obj, dict) else {}


def infer_cold_object(data_dir: Path, requested: str | None) -> str:
    if requested and requested != "auto":
        return requested
    para = load_convert(data_dir)
    cold_object = para.get("cold_object")
    if cold_object in {"item", "user"}:
        return cold_object
    if (data_dir / "cold_item_test.csv").exists():
        return "item"
    if (data_dir / "cold_user_test.csv").exists():
        return "user"
    raise ValueError(f"Could not infer cold_object from {data_dir}")


def load_mapped_embeddings(args, data_dir: Path, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    user_emb_path = Path(args.user_llm_emb) if args.user_llm_emb else data_dir / "llm_user_content_emb.pt"
    item_emb_path = Path(args.item_llm_emb) if args.item_llm_emb else data_dir / "llm_item_content_emb.pt"
    head_path = Path(args.llama_head) if args.llama_head else data_dir / "llama_head.bin"

    user_origin = torch.load(user_emb_path, map_location="cpu").float().to(device)
    item_origin = torch.load(item_emb_path, map_location="cpu").float().to(device)
    if user_origin.shape[1] != item_origin.shape[1]:
        raise ValueError(f"LLM embedding dim mismatch: user={tuple(user_origin.shape)}, item={tuple(item_origin.shape)}")

    model = LlamaHead(user_origin.shape[1], args.hidden_size, args.output_size).to(device)
    model.load_state_dict(torch.load(head_path, map_location="cpu"))
    model.eval()
    with torch.no_grad():
        mapped_user = model.user_mlp(user_origin)
        mapped_item = model.item_mlp(item_origin)
        if args.normalize:
            mapped_user = F.normalize(mapped_user, p=2, dim=1)
            mapped_item = F.normalize(mapped_item, p=2, dim=1)
    return mapped_user, mapped_item


def load_filter_embeddings(args, data_dir: Path, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    parts: list[tuple[torch.Tensor, torch.Tensor]] = []
    if args.filter_source in {"llm", "hybrid"}:
        parts.append(load_mapped_embeddings(args, data_dir, device))
    if args.filter_source in {"aldi", "hybrid"}:
        user_aldi_path = Path(args.user_aldi_emb) if args.user_aldi_emb else data_dir / "ALDI_user_emb.pt"
        item_aldi_path = Path(args.item_aldi_emb) if args.item_aldi_emb else data_dir / "ALDI_item_emb.pt"
        aldi_user = torch.load(user_aldi_path, map_location="cpu").float().to(device)
        aldi_item = torch.load(item_aldi_path, map_location="cpu").float().to(device)
        if args.normalize_aldi:
            aldi_user = F.normalize(aldi_user, p=2, dim=1)
            aldi_item = F.normalize(aldi_item, p=2, dim=1)
        parts.append((aldi_user, aldi_item))
    if not parts:
        raise ValueError(f"Unsupported filter_source={args.filter_source!r}")
    if len(parts) == 1:
        return parts[0]
    return torch.cat([part[0] for part in parts], dim=1), torch.cat([part[1] for part in parts], dim=1)


def select_topk_from_scores(
    scores: torch.Tensor,
    candidate_ids: torch.Tensor,
    observed: set[int],
    topk: int,
) -> tuple[list[int], list[float]]:
    if observed:
        blocked = torch.tensor(
            [idx for idx, entity_id in enumerate(candidate_ids.tolist()) if int(entity_id) in observed],
            dtype=torch.long,
            device=scores.device,
        )
        if len(blocked):
            scores = scores.clone()
            scores[blocked] = -float("inf")
    k = min(topk, int(torch.isfinite(scores).sum().item()))
    if k <= 0:
        return [], []
    values, local = torch.topk(scores, k=k)
    selected = candidate_ids[local].detach().cpu().tolist()
    return [int(x) for x in selected], [float(x) for x in values.detach().cpu().tolist()]


def build_item_cold(
    mapped_user: torch.Tensor,
    mapped_item: torch.Tensor,
    targets: dict[int, str],
    observed: dict[int, set[int]],
    topk: int,
    batch_size: int,
) -> pd.DataFrame:
    device = mapped_user.device
    candidate_users = torch.arange(mapped_user.shape[0], dtype=torch.long, device=device)
    target_items = list(targets)
    rows = []
    for start in tqdm(range(0, len(target_items), batch_size), desc="Paper-style item-cold candidates"):
        batch_items = target_items[start : start + batch_size]
        item_tensor = torch.as_tensor(batch_items, dtype=torch.long, device=device)
        score_batch = mapped_item[item_tensor] @ mapped_user.t()
        for row_idx, item in enumerate(batch_items):
            users, scores = select_topk_from_scores(score_batch[row_idx], candidate_users, observed.get(int(item), set()), topk)
            for user, score in zip(users, scores):
                rows.append(
                    {
                        "user": int(user),
                        "item": int(item),
                        "entity_type": targets[int(item)],
                        "candidate_score": float(score),
                    }
                )
    return pd.DataFrame(rows, columns=["user", "item", "entity_type", "candidate_score"])


def build_user_cold(
    mapped_user: torch.Tensor,
    mapped_item: torch.Tensor,
    targets: dict[int, str],
    observed: dict[int, set[int]],
    data_dir: Path,
    topk: int,
    batch_size: int,
) -> pd.DataFrame:
    train = read_pairs(data_dir / "warm_emb.csv")
    if len(train) == 0:
        raise ValueError("warm_emb.csv is required to define visible candidate items for cold-user filtering.")
    visible_items = sorted(train["item"].unique().tolist())
    candidate_items = torch.as_tensor(visible_items, dtype=torch.long, device=mapped_user.device)
    visible_item_emb = mapped_item[candidate_items]
    target_users = list(targets)
    rows = []
    for start in tqdm(range(0, len(target_users), batch_size), desc="Paper-style user-cold candidates"):
        batch_users = target_users[start : start + batch_size]
        user_tensor = torch.as_tensor(batch_users, dtype=torch.long, device=mapped_user.device)
        score_batch = mapped_user[user_tensor] @ visible_item_emb.t()
        for row_idx, user in enumerate(batch_users):
            items, scores = select_topk_from_scores(score_batch[row_idx], candidate_items, observed.get(int(user), set()), topk)
            for item, score in zip(items, scores):
                rows.append(
                    {
                        "user": int(user),
                        "item": int(item),
                        "entity_type": targets[int(user)],
                        "candidate_score": float(score),
                    }
                )
    return pd.DataFrame(rows, columns=["user", "item", "entity_type", "candidate_score"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Build top-K candidates with the paper-style filter.")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--cold_object", choices=["auto", "item", "user"], default="auto")
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--hidden_size", type=int, default=1024)
    parser.add_argument("--output_size", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--filter_source", choices=["llm", "aldi", "hybrid"], default="llm")
    parser.add_argument("--normalize_aldi", action="store_true")
    parser.add_argument("--user_llm_emb", default=None)
    parser.add_argument("--item_llm_emb", default=None)
    parser.add_argument("--llama_head", default=None)
    parser.add_argument("--user_aldi_emb", default=None)
    parser.add_argument("--item_aldi_emb", default=None)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--diagnostics_json", default=None)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    cold_object = infer_cold_object(data_dir, args.cold_object)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    targets = load_targets(data_dir, cold_object)
    if not targets:
        raise ValueError(f"No cold/warm-up targets found in {data_dir}")
    observed = load_observed(data_dir, cold_object)
    mapped_user, mapped_item = load_filter_embeddings(args, data_dir, device)

    if cold_object == "item":
        out = build_item_cold(mapped_user, mapped_item, targets, observed, args.topk, args.batch_size)
    else:
        out = build_user_cold(mapped_user, mapped_item, targets, observed, data_dir, args.topk, args.batch_size)

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)

    diagnostics = {
        "method": "paper_style_llm_mapped_embedding_topk" if args.filter_source == "llm" else f"{args.filter_source}_embedding_topk",
        "data_dir": str(data_dir.resolve()),
        "cold_object": cold_object,
        "topk": int(args.topk),
        "filter_source": args.filter_source,
        "normalize": bool(args.normalize),
        "targets": int(len(targets)),
        "rows": int(len(out)),
        "unique_users": int(out["user"].nunique()) if len(out) else 0,
        "unique_items": int(out["item"].nunique()) if len(out) else 0,
        "entity_type_counts": out["entity_type"].value_counts().to_dict() if len(out) else {},
        "output_csv": str(output_csv.resolve()),
    }
    print(json.dumps(diagnostics, indent=2, ensure_ascii=False))
    if args.diagnostics_json:
        Path(args.diagnostics_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.diagnostics_json).write_text(json.dumps(diagnostics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
