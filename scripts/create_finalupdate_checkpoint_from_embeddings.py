#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a FinalUpdate checkpoint from exported user/item embedding numpy arrays."
    )
    parser.add_argument("--user_emb", required=True)
    parser.add_argument("--item_emb", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--meta_output", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--method", default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    user_path = Path(args.user_emb)
    item_path = Path(args.item_emb)
    output = Path(args.output)
    meta_output = Path(args.meta_output) if args.meta_output else output.with_suffix(output.suffix + ".meta.json")

    def load_embedding(path: Path) -> torch.Tensor:
        if path.suffix == ".npy":
            import numpy as np

            return torch.from_numpy(np.load(path).astype(np.float32, copy=False))
        emb = torch.load(path, map_location="cpu")
        if not torch.is_tensor(emb):
            raise TypeError(f"Expected tensor or .npy array at {path}, got {type(emb)!r}")
        return emb.detach().cpu().float()

    user_emb = load_embedding(user_path)
    item_emb = load_embedding(item_path)
    if user_emb.ndim != 2 or item_emb.ndim != 2:
        raise ValueError(f"Expected 2D embeddings, got user={user_emb.shape} item={item_emb.shape}")
    if user_emb.shape[1] != item_emb.shape[1]:
        raise ValueError(f"Embedding dimensions differ: user={user_emb.shape} item={item_emb.shape}")

    state = {
        "embedding_user.weight": user_emb.clone(),
        "embedding_item.weight": item_emb.clone(),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, output)

    meta = {
        "source_user_emb": str(user_path.resolve()),
        "source_item_emb": str(item_path.resolve()),
        "output_checkpoint": str(output.resolve()),
        "user_emb_shape": list(user_emb.shape),
        "item_emb_shape": list(item_emb.shape),
        "dtype": "float32",
        "model": args.model,
        "method": args.method,
        "seed": args.seed,
    }
    meta_output.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2), flush=True)


if __name__ == "__main__":
    main()
