#!/usr/bin/env python3
import argparse
from pathlib import Path

import pandas as pd


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def add_probability(df: pd.DataFrame, value: float) -> pd.DataFrame:
    out = df.copy()
    if "probability" not in out.columns:
        out["probability"] = value
    return out[["user", "item", "entity_type", "probability"]]


def random_like_reference(top20: pd.DataFrame, reference: pd.DataFrame, seed: int) -> pd.DataFrame:
    pieces = []
    for entity_type, ref_group in reference.groupby("entity_type"):
        candidates = top20[top20["entity_type"] == entity_type]
        n = min(len(ref_group), len(candidates))
        if n == 0:
            continue
        pieces.append(candidates.sample(n=n, random_state=seed).copy())
    if not pieces:
        return pd.DataFrame(columns=["user", "item", "entity_type", "probability"])
    return add_probability(pd.concat(pieces, ignore_index=True), 1.0)


def main():
    parser = argparse.ArgumentParser(description="Build revision ablation interaction CSVs.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--source_experiment", default="experiments/multiseed/CiteULike_item")
    parser.add_argument("--reference_experiment", default="experiments/multiseed/CiteULike_item")
    parser.add_argument("--out_root", default="experiments/revision_ablation_inputs/CiteULike_item")
    args = parser.parse_args()

    root = Path(args.root)
    seed_dir = root / args.source_experiment / f"seed_{args.seed}"
    ref_dir = root / args.reference_experiment / f"seed_{args.seed}"
    out_dir = root / args.out_root / f"seed_{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    top20 = read_csv(seed_dir / "top20.csv")
    reference = read_csv(ref_dir / "predicted_cold_item_interaction.csv")

    embsim = add_probability(top20, 1.0)
    random_sim = random_like_reference(top20, reference, args.seed)

    embsim_path = out_dir / f"embsim_only_seed{args.seed}.csv"
    random_path = out_dir / f"random_sim_seed{args.seed}.csv"
    embsim.to_csv(embsim_path, index=False)
    random_sim.to_csv(random_path, index=False)

    print(f"Saved {len(embsim)} EmbSim-only interactions to {embsim_path}")
    print(f"Saved {len(random_sim)} RandomSim interactions to {random_path}")


if __name__ == "__main__":
    main()
