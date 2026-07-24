#!/usr/bin/env python3
"""Export raw FinalUpdate checkpoint scores for later cross-runroot blending."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch


SPLITS = [
    ("strict_cold", 0),
    ("warmup", 1),
    ("warm", 2),
    ("overall", 3),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runroot", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--extended_file", required=True)
    parser.add_argument("--output_npz", required=True)
    parser.add_argument("--dataset", default="CiteULike")
    parser.add_argument("--model", default="lgn")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--graph_cache", default="")
    parser.add_argument("--rwft_weighted", type=int, default=1)
    parser.add_argument("--rwft_beta", type=float, default=1.5)
    parser.add_argument("--caga_gamma", type=float, default=0.0)
    parser.add_argument("--caga_k0", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=512)
    return parser.parse_args()


def configure(args: argparse.Namespace) -> None:
    finalupdate_dir = Path(args.runroot).resolve() / "FinalUpdate"
    os.chdir(finalupdate_dir)
    sys.path.insert(0, str(finalupdate_dir))
    sys.argv = [
        "export_checkpoint_scores_for_blend.py",
        "--dataset",
        args.dataset,
        "--seed",
        str(args.seed),
        "--model",
        args.model,
        "--load",
        "0",
        "--epochs",
        "0",
        "--testbatch",
        str(args.batch_size),
        "--extended_file",
        str(Path(args.extended_file).resolve()),
        "--rwft_weighted",
        str(args.rwft_weighted),
        "--rwft_beta",
        str(args.rwft_beta),
        "--caga_gamma",
        str(args.caga_gamma),
        "--caga_k0",
        str(args.caga_k0),
        "--file_name",
        "export_checkpoint_scores_for_blend",
    ]
    if args.graph_cache:
        sys.argv.extend(["--graph_cache", args.graph_cache])


def as_array(value) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value.astype(np.int64, copy=False).reshape(-1)
    if isinstance(value, (list, tuple, set)):
        return np.asarray(list(value), dtype=np.int64).reshape(-1)
    return np.asarray([int(value)], dtype=np.int64)


def combine_masks(*arrays) -> np.ndarray:
    valid = [as_array(array) for array in arrays if array is not None and len(array) > 0]
    if not valid:
        return np.asarray([], dtype=np.int64)
    return np.unique(np.concatenate(valid))


def split_bundle(dataset, split_idx: int):
    neighbors = dataset.test_user_nb()[split_idx]
    users = as_array(dataset.test_user()[split_idx])
    return neighbors, users


def load_model(register, world, dataset, checkpoint: str):
    recmodel = register.MODELS[world.model_name](world.config, dataset).to(world.device)
    recmodel.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    recmodel.eval()
    return recmodel


def precompute_model_scores(model, users: np.ndarray, world, batch_size: int) -> np.ndarray:
    chunks = []
    with torch.no_grad():
        for begin in range(0, len(users), batch_size):
            end = min(begin + batch_size, len(users))
            batch = torch.as_tensor(users[begin:end], dtype=torch.long, device=world.device)
            scores = model.getUsersRating(batch).detach().cpu().numpy().astype(np.float32)
            chunks.append(scores)
            print(f"[PRECOMPUTE] users={end}/{len(users)}", flush=True)
    return np.vstack(chunks)


def split_eval_data(dataset, split_name: str, split_idx: int, masked_items: np.ndarray):
    neighbors, users = split_bundle(dataset, split_idx)
    pos_user_nb = dataset.para_dict["pos_user_nb"]
    gt_offsets = [0]
    gt_items = []
    exclude_rows = []
    exclude_cols = []
    for row_idx, user in enumerate(users.tolist()):
        gt = as_array(neighbors[int(user)])
        gt_set = set(gt.tolist())
        gt_items.extend(gt.tolist())
        gt_offsets.append(len(gt_items))
        positives = as_array(pos_user_nb[int(user)])
        if positives.size:
            excluded = [int(item) for item in positives.tolist() if int(item) not in gt_set]
            if excluded:
                exclude_rows.extend([row_idx] * len(excluded))
                exclude_cols.extend(excluded)
    print(
        f"[SPLIT] split={split_name} users={len(users)} "
        f"gt_items={len(gt_items)} exclude={len(exclude_rows)} masked={len(masked_items)}",
        flush=True,
    )
    return {
        "users": users.astype(np.int64),
        "gt_offsets": np.asarray(gt_offsets, dtype=np.int64),
        "gt_items": np.asarray(gt_items, dtype=np.int64),
        "exclude_rows": np.asarray(exclude_rows, dtype=np.int64),
        "exclude_cols": np.asarray(exclude_cols, dtype=np.int64),
        "masked_items": np.asarray(masked_items, dtype=np.int64),
    }


def main() -> None:
    args = parse_args()
    os.environ["PROBLLM_EVAL_CANDIDATE_MODE"] = "group"
    os.environ["PROBLLM_SCORE_PRIOR_FILE"] = ""
    os.environ["PROBLLM_SCORE_PRIOR_ALPHA"] = "0"
    configure(args)

    import register  # noqa: WPS433
    import world  # noqa: WPS433

    dataset = register.dataset
    model = load_model(register, world, dataset, args.checkpoint)

    warm_items, strict_items, warmup_items = dataset.eval_item_groups()
    warm_items = as_array(warm_items)
    strict_items = as_array(strict_items)
    warmup_items = as_array(warmup_items)
    group_id = np.zeros(dataset.m_items, dtype=np.int8)
    group_id[warm_items] = 3
    group_id[strict_items] = 1
    group_id[warmup_items] = 2
    split_masks = {
        "strict_cold": combine_masks(warm_items, warmup_items),
        "warmup": combine_masks(warm_items, strict_items),
        "warm": combine_masks(strict_items, warmup_items),
        "overall": np.asarray([], dtype=np.int64),
    }

    arrays = {
        "group_id": group_id,
        "split_names": np.asarray([name for name, _idx in SPLITS], dtype=object),
    }
    for split_name, split_idx in SPLITS:
        data = split_eval_data(dataset, split_name, split_idx, split_masks[split_name])
        scores = precompute_model_scores(model, data["users"], world, args.batch_size)
        arrays[f"{split_name}_scores"] = scores
        for key, value in data.items():
            arrays[f"{split_name}_{key}"] = value

    meta = {
        "runroot": str(Path(args.runroot).resolve()),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "extended_file": str(Path(args.extended_file).resolve()),
        "dataset": args.dataset,
        "model": args.model,
        "seed": args.seed,
        "graph_cache": args.graph_cache,
        "rwft_weighted": args.rwft_weighted,
        "rwft_beta": args.rwft_beta,
        "caga_gamma": args.caga_gamma,
        "caga_k0": args.caga_k0,
    }
    arrays["meta_json"] = np.asarray(json.dumps(meta, sort_keys=True), dtype=object)

    out = Path(args.output_npz)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **arrays)
    print(f"[EXPORT-CHECKPOINT-SCORES] wrote={out} meta={json.dumps(meta, sort_keys=True)}")


if __name__ == "__main__":
    main()
