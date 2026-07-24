#!/usr/bin/env python3
"""Export a warm-item score prior from a FinalUpdate checkpoint.

The output CSV uses the same columns consumed by Procedure.score_prior_matrix:
user,item,probability.  It is intended for fast 0-epoch evaluation sweeps.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runroot", required=True, help="Isolated FinalUpdate run root.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--extended_file", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--dataset", default="CiteULike")
    parser.add_argument("--model", default="lgn")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--graph_cache", default="")
    parser.add_argument("--rwft_weighted", type=int, default=1)
    parser.add_argument("--rwft_beta", type=float, default=1.5)
    parser.add_argument("--caga_gamma", type=float, default=0.0)
    parser.add_argument("--caga_k0", type=int, default=5)
    parser.add_argument("--sim_prob_column", default="probability")
    parser.add_argument("--testbatch", type=int, default=512)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument(
        "--prob_mode",
        choices=("rank_linear", "rank_recip", "constant", "score_minmax"),
        default="rank_linear",
    )
    parser.add_argument("--rank_floor", type=float, default=0.5)
    parser.add_argument("--mask_train", type=int, default=1)
    return parser.parse_args()


def configure_finalupdate(args: argparse.Namespace) -> None:
    runroot = Path(args.runroot).resolve()
    finalupdate_dir = runroot / "FinalUpdate"
    if not finalupdate_dir.is_dir():
        raise FileNotFoundError(f"FinalUpdate directory not found: {finalupdate_dir}")

    os.chdir(finalupdate_dir)
    sys.path.insert(0, str(finalupdate_dir))
    sys.argv = [
        "export_finalupdate_warm_prior.py",
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
        str(args.testbatch),
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
        "--sim_prob_column",
        args.sim_prob_column,
        "--file_name",
        "export_warm_prior",
    ]
    if args.graph_cache:
        sys.argv.extend(["--graph_cache", args.graph_cache])


def rank_probabilities(n: int, mode: str, floor: float, scores: torch.Tensor) -> np.ndarray:
    if n <= 0:
        return np.empty((0,), dtype=np.float32)
    if mode == "constant":
        return np.ones((n,), dtype=np.float32)
    if mode == "rank_recip":
        denom = np.arange(1, n + 1, dtype=np.float32)
        return (1.0 / denom).astype(np.float32)
    if mode == "score_minmax":
        values = scores.detach().float().cpu().numpy()
        lo = float(values.min())
        hi = float(values.max())
        if hi <= lo:
            return np.ones((n,), dtype=np.float32)
        return (floor + (1.0 - floor) * (values - lo) / (hi - lo)).astype(np.float32)
    if n == 1:
        return np.ones((1,), dtype=np.float32)
    return np.linspace(1.0, floor, n, dtype=np.float32)


def main() -> None:
    args = parse_args()
    configure_finalupdate(args)

    import register  # noqa: WPS433
    import world  # noqa: WPS433

    dataset = register.dataset
    warm_item, _strict_cold_item, _warmup_item = dataset.eval_item_groups()
    warm_item = np.asarray(warm_item, dtype=np.int64)
    if len(warm_item) == 0:
        raise RuntimeError("No warm items found in this split.")

    recmodel = register.MODELS[world.model_name](world.config, dataset).to(world.device)
    state = torch.load(args.checkpoint, map_location="cpu")
    recmodel.load_state_dict(state)
    recmodel.eval()

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    top_k = min(args.top_k, len(warm_item))
    warm_tensor = torch.as_tensor(warm_item, dtype=torch.long, device=world.device)
    train_pos = dataset.allPos if args.mask_train else [np.array([], dtype=np.int64) for _ in range(dataset.n_users)]

    rows = 0
    with output_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["user", "item", "probability"])
        for start in range(0, dataset.n_users, args.testbatch):
            end = min(start + args.testbatch, dataset.n_users)
            users = torch.arange(start, end, dtype=torch.long, device=world.device)
            with torch.no_grad():
                scores = recmodel.getUsersRating(users)[:, warm_tensor]
            if args.mask_train:
                for offset, user in enumerate(range(start, end)):
                    positives = np.asarray(train_pos[user], dtype=np.int64)
                    if positives.size == 0:
                        continue
                    local = np.nonzero(np.isin(warm_item, positives, assume_unique=False))[0]
                    if local.size > 0:
                        scores[offset, torch.as_tensor(local, dtype=torch.long, device=world.device)] = -1e10
            values, indices = torch.topk(scores, k=top_k, dim=1)
            values_cpu = values.detach().cpu()
            indices_cpu = indices.detach().cpu().numpy()
            for offset, user in enumerate(range(start, end)):
                probs = rank_probabilities(top_k, args.prob_mode, args.rank_floor, values_cpu[offset])
                items = warm_item[indices_cpu[offset]]
                for item, prob in zip(items, probs):
                    writer.writerow([user, int(item), f"{float(prob):.8f}"])
                    rows += 1

    print(
        f"[WARM-PRIOR] wrote={output_csv} rows={rows} users={dataset.n_users} "
        f"warm_items={len(warm_item)} top_k={top_k} mode={args.prob_mode}"
    )


if __name__ == "__main__":
    main()
