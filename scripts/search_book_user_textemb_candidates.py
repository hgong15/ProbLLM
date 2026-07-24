#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm


def parse_ints(text: str) -> list[int]:
    return sorted({int(piece) for piece in text.split(",") if piece.strip()})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Search Book-Crossing user-cold TextEmb candidate generators. "
            "The search ranks candidates by validation hit metrics; test metrics are reported "
            "only for diagnosis."
        )
    )
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--embedding_path", type=Path, default=None)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--neighbor_ks", default="10,20,50,100,200,500")
    parser.add_argument("--pool_m", type=int, default=50)
    parser.add_argument("--strict_k", type=int, default=10)
    parser.add_argument("--warmup_k", type=int, default=20)
    parser.add_argument("--sim_batch_size", type=int, default=512)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--limit_users", type=int, default=0)
    parser.add_argument(
        "--save_variant",
        default="",
        help="Optional exact variant name from summary.csv to save as selected CSV.",
    )
    parser.add_argument(
        "--save_only",
        action="store_true",
        help="When --save_variant is set, evaluate and save only that variant.",
    )
    parser.add_argument("--save_csv", type=Path, default=None)
    return parser.parse_args()


def read_pairs(path: Path, empty_ok: bool = True) -> pd.DataFrame:
    if not path.exists():
        if empty_ok:
            return pd.DataFrame(columns=["user", "item"])
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    if df.empty:
        return pd.DataFrame(columns=["user", "item"])
    return df[["user", "item"]].astype({"user": int, "item": int})


def load_embedding(path: Path) -> torch.Tensor:
    if path.suffix == ".pt":
        emb = torch.load(path, map_location="cpu").float()
    elif path.suffix == ".npy":
        emb = torch.from_numpy(np.asarray(np.load(path, mmap_mode="r"), dtype=np.float32))
    else:
        raise ValueError(f"Unsupported embedding format: {path}")
    if emb.ndim != 2:
        raise ValueError(f"Expected 2D embedding at {path}, got shape={tuple(emb.shape)}")
    return F.normalize(emb.float(), p=2, dim=1)


def target_user_types(data_dir: Path) -> dict[int, str]:
    targets: dict[int, str] = {}
    for name in ("cold_user_val.csv", "cold_user_test.csv"):
        for user in read_pairs(data_dir / name)["user"].unique().tolist():
            targets[int(user)] = "strict_cold"
    for name in ("warmup_val.csv", "warmup_test.csv"):
        for user in read_pairs(data_dir / name)["user"].unique().tolist():
            targets[int(user)] = "warmup"
    return targets


def observed_by_user(data_dir: Path) -> dict[int, set[int]]:
    observed: dict[int, set[int]] = defaultdict(set)
    for name in ("warm_emb.csv", "warm_train.csv", "warmup_support.csv"):
        for user, item in read_pairs(data_dir / name).itertuples(index=False):
            observed[int(user)].add(int(item))
    return observed


def build_truth(data_dir: Path) -> dict[str, dict[int, set[int]]]:
    truth = {}
    for split, filename in {
        "strict_val": "cold_user_val.csv",
        "strict_test": "cold_user_test.csv",
        "warmup_val": "warmup_val.csv",
        "warmup_test": "warmup_test.csv",
    }.items():
        by_user: dict[int, set[int]] = defaultdict(set)
        for user, item in read_pairs(data_dir / filename).itertuples(index=False):
            by_user[int(user)].add(int(item))
        truth[split] = by_user
    return truth


def build_train_indexes(train_df: pd.DataFrame):
    items_by_user: dict[int, list[int]] = defaultdict(list)
    item_counter: Counter[int] = Counter()
    for user, item in train_df[["user", "item"]].itertuples(index=False):
        user_id = int(user)
        item_id = int(item)
        items_by_user[user_id].append(item_id)
        item_counter[item_id] += 1
    popular_items = [item for item, _ in item_counter.most_common()]
    return items_by_user, item_counter, popular_items


@dataclass
class HitCounter:
    hits: int = 0
    total: int = 0
    user_hit: int = 0
    users: int = 0

    def add(self, truth_items: set[int], selected: set[int]) -> None:
        if not truth_items:
            return
        found = len(truth_items & selected)
        self.hits += found
        self.total += len(truth_items)
        self.user_hit += int(found > 0)
        self.users += 1

    def to_dict(self, prefix: str) -> dict[str, float | int]:
        return {
            f"{prefix}_hits": int(self.hits),
            f"{prefix}_total": int(self.total),
            f"{prefix}_recall": float(self.hits / self.total) if self.total else 0.0,
            f"{prefix}_user_hit_rate": float(self.user_hit / self.users) if self.users else 0.0,
            f"{prefix}_users": int(self.users),
        }


@dataclass
class VariantStats:
    name: str
    neighbor_k: int
    policy: str
    budget: str
    counters: dict[str, HitCounter] = field(
        default_factory=lambda: {
            "strict_val": HitCounter(),
            "strict_test": HitCounter(),
            "warmup_val": HitCounter(),
            "warmup_test": HitCounter(),
        }
    )
    rows: int = 0
    selected_items: set[int] = field(default_factory=set)

    def add_user(
        self,
        user: int,
        entity_type: str,
        selected_items: list[int],
        truth: dict[str, dict[int, set[int]]],
    ) -> None:
        selected = set(selected_items)
        self.rows += len(selected_items)
        self.selected_items.update(selected)
        if entity_type == "strict_cold":
            self.counters["strict_val"].add(truth["strict_val"].get(user, set()), selected)
            self.counters["strict_test"].add(truth["strict_test"].get(user, set()), selected)
        elif entity_type == "warmup":
            self.counters["warmup_val"].add(truth["warmup_val"].get(user, set()), selected)
            self.counters["warmup_test"].add(truth["warmup_test"].get(user, set()), selected)

    def row(self) -> dict[str, float | int | str]:
        out: dict[str, float | int | str] = {
            "variant": self.name,
            "neighbor_k": self.neighbor_k,
            "policy": self.policy,
            "budget": self.budget,
            "rows": int(self.rows),
            "candidate_items": int(len(self.selected_items)),
        }
        for key, counter in self.counters.items():
            out.update(counter.to_dict(key))
        strict = float(out.get("strict_val_recall", 0.0))
        warmup = float(out.get("warmup_val_recall", 0.0))
        out["val_macro_recall"] = 0.5 * (strict + warmup)
        out["val_weighted_recall"] = (
            float(out.get("strict_val_hits", 0)) + float(out.get("warmup_val_hits", 0))
        ) / max(float(out.get("strict_val_total", 0)) + float(out.get("warmup_val_total", 0)), 1.0)
        return out


def score_items(
    features: dict[int, list[float]],
    item_pop: Counter[int],
    policy: str,
) -> list[tuple[int, float]]:
    scored: list[tuple[int, float]] = []
    for item, values in features.items():
        sum_sim, max_sim, count, rank_decay, sum_sq, sum_sqrt = values
        pop = float(item_pop.get(item, 1))
        if policy == "sum":
            score = sum_sim
        elif policy == "max":
            score = max_sim
        elif policy == "avg":
            score = sum_sim / max(count, 1.0)
        elif policy == "count":
            score = count + 1e-6 * sum_sim
        elif policy == "rank_decay":
            score = rank_decay
        elif policy == "sum_sq":
            score = sum_sq
        elif policy == "sum_sqrt":
            score = sum_sqrt
        elif policy == "sum_count_boost":
            score = sum_sim * math.log1p(count)
        elif policy == "sum_idf025":
            score = sum_sim / (math.log1p(pop) ** 0.25)
        elif policy == "sum_idf050":
            score = sum_sim / (math.log1p(pop) ** 0.50)
        elif policy == "sum_idf100":
            score = sum_sim / max(math.log1p(pop), 1e-12)
        elif policy == "sum_pop025":
            score = sum_sim * (math.log1p(pop) ** 0.25)
        elif policy == "sum_pop050":
            score = sum_sim * (math.log1p(pop) ** 0.50)
        else:
            raise ValueError(f"Unknown policy: {policy}")
        scored.append((item, float(score)))
    return sorted(scored, key=lambda row: (-row[1], row[0]))


def normalize_scores(rows: list[tuple[int, float]]) -> list[tuple[int, float]]:
    if not rows:
        return []
    scores = np.asarray([score for _, score in rows], dtype=np.float32)
    lo = float(scores.min())
    hi = float(scores.max())
    if hi > lo:
        probs = (scores - lo) / (hi - lo)
    else:
        probs = np.ones_like(scores, dtype=np.float32)
    return [(item, float(prob)) for (item, _), prob in zip(rows, probs.tolist())]


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir
    args.output_dir.mkdir(parents=True, exist_ok=True)
    emb_path = args.embedding_path or data_dir / "book-crossing_user_content.npy"
    user_emb = load_embedding(emb_path)
    train_df = read_pairs(data_dir / "warm_emb.csv", empty_ok=False)
    items_by_user, item_pop, popular_items = build_train_indexes(train_df)
    blocked_by_user = observed_by_user(data_dir)
    targets = target_user_types(data_dir)
    truth = build_truth(data_dir)

    target_users = sorted(targets)
    if args.limit_users > 0:
        target_users = target_users[: args.limit_users]

    neighbor_ks = parse_ints(args.neighbor_ks)
    if not neighbor_ks:
        raise ValueError("neighbor_ks is empty")
    policies = [
        "sum",
        "max",
        "avg",
        "count",
        "rank_decay",
        "sum_sq",
        "sum_sqrt",
        "sum_count_boost",
        "sum_idf025",
        "sum_idf050",
        "sum_idf100",
        "sum_pop025",
        "sum_pop050",
    ]
    save_spec = None
    if args.save_variant:
        for k in neighbor_ks:
            prefix = f"textemb_neighbor_k{k}_"
            if not args.save_variant.startswith(prefix):
                continue
            suffix = args.save_variant[len(prefix) :]
            budget_tag = f"_budget_s{args.strict_k}_w{args.warmup_k}"
            pool_tag = f"_pool{args.pool_m}"
            if suffix.endswith(budget_tag):
                policy = suffix[: -len(budget_tag)]
                budget_name = f"budget_s{args.strict_k}_w{args.warmup_k}"
            elif suffix.endswith(pool_tag):
                policy = suffix[: -len(pool_tag)]
                budget_name = f"pool{args.pool_m}"
            else:
                continue
            if policy not in policies:
                raise ValueError(f"save_variant policy not recognized: {policy}")
            save_spec = {"k": k, "policy": policy, "budget": budget_name}
            break
        if save_spec is None:
            raise ValueError(f"save_variant not recognized by current config: {args.save_variant}")
        if args.save_only:
            neighbor_ks = [int(save_spec["k"])]
            policies = [str(save_spec["policy"])]
    max_k = max(neighbor_ks)

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    warm_users_cpu = torch.as_tensor(sorted(items_by_user), dtype=torch.long)
    warm_emb = user_emb[warm_users_cpu].to(device)
    source_pos_by_user = {int(user): idx for idx, user in enumerate(warm_users_cpu.tolist())}
    take = min(max_k, int(warm_users_cpu.numel()))

    stats: dict[str, VariantStats] = {}
    save_rows: list[dict[str, int | str | float]] = []

    def get_stats(k: int, policy: str, budget: str) -> VariantStats:
        name = f"textemb_neighbor_k{k}_{policy}_{budget}"
        if name not in stats:
            stats[name] = VariantStats(name=name, neighbor_k=k, policy=policy, budget=budget)
        return stats[name]

    for start in tqdm(range(0, len(target_users), args.sim_batch_size), desc="Searching TextEmb candidates"):
        batch_users = target_users[start : start + args.sim_batch_size]
        batch_tensor = torch.as_tensor(batch_users, dtype=torch.long)
        batch_emb = user_emb[batch_tensor].to(device)
        sims = batch_emb @ warm_emb.t()
        for row_idx, user in enumerate(batch_users):
            same_pos = source_pos_by_user.get(int(user))
            if same_pos is not None:
                sims[row_idx, same_pos] = -torch.inf
        values, local_indices = torch.topk(sims, k=take, dim=1)
        values_np = values.detach().cpu().numpy()
        local_np = local_indices.detach().cpu().numpy()

        for row_idx, user in enumerate(batch_users):
            user_id = int(user)
            entity_type = targets[user_id]
            budget_k = args.strict_k if entity_type == "strict_cold" else args.warmup_k
            blocked = blocked_by_user.get(user_id, set())
            features: dict[int, list[float]] = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            next_k_idx = 0
            per_policy_ranked: dict[str, list[tuple[int, float]]] = {}

            for rank0, (sim, local_idx) in enumerate(zip(values_np[row_idx].tolist(), local_np[row_idx].tolist())):
                if not np.isfinite(sim):
                    continue
                sim = max(float(sim), 0.0)
                if sim <= 0.0:
                    continue
                warm_user = int(warm_users_cpu[int(local_idx)])
                rank_weight = sim / math.log2(rank0 + 2.0)
                for item in items_by_user.get(warm_user, []):
                    item = int(item)
                    if item in blocked:
                        continue
                    vals = features[item]
                    vals[0] += sim
                    vals[1] = max(vals[1], sim)
                    vals[2] += 1.0
                    vals[3] += rank_weight
                    vals[4] += sim * sim
                    vals[5] += math.sqrt(sim)

                current_k = rank0 + 1
                while next_k_idx < len(neighbor_ks) and current_k >= neighbor_ks[next_k_idx]:
                    k = neighbor_ks[next_k_idx]
                    next_k_idx += 1
                    per_policy_ranked.clear()
                    for policy in policies:
                        ranked = score_items(features, item_pop, policy)
                        if len(ranked) < args.pool_m:
                            selected_items = {item for item, _ in ranked}
                            ranked = list(ranked)
                            for item in popular_items:
                                if item in blocked or item in selected_items:
                                    continue
                                ranked.append((int(item), 0.0))
                                selected_items.add(int(item))
                                if len(ranked) >= args.pool_m:
                                    break
                        per_policy_ranked[policy] = ranked
                        pool = [item for item, _ in ranked[: args.pool_m]]
                        budget = [item for item, _ in ranked[:budget_k]]
                        get_stats(k, policy, f"pool{args.pool_m}").add_user(user_id, entity_type, pool, truth)
                        get_stats(k, policy, f"budget_s{args.strict_k}_w{args.warmup_k}").add_user(
                            user_id, entity_type, budget, truth
                        )

            if args.save_variant:
                k = int(save_spec["k"])
                policy = str(save_spec["policy"])
                out_k = args.pool_m if save_spec["budget"] == f"pool{args.pool_m}" else budget_k
                # Re-score from the full prefix of k neighbors for the requested variant.
                features_save: dict[int, list[float]] = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                for rank0, (sim, local_idx) in enumerate(
                    zip(values_np[row_idx].tolist()[:k], local_np[row_idx].tolist()[:k])
                ):
                    if not np.isfinite(sim):
                        continue
                    sim = max(float(sim), 0.0)
                    if sim <= 0.0:
                        continue
                    warm_user = int(warm_users_cpu[int(local_idx)])
                    rank_weight = sim / math.log2(rank0 + 2.0)
                    for item in items_by_user.get(warm_user, []):
                        item = int(item)
                        if item in blocked:
                            continue
                        vals = features_save[item]
                        vals[0] += sim
                        vals[1] = max(vals[1], sim)
                        vals[2] += 1.0
                        vals[3] += rank_weight
                        vals[4] += sim * sim
                        vals[5] += math.sqrt(sim)
                ranked = score_items(features_save, item_pop, policy)
                if len(ranked) < out_k:
                    selected_items = {item for item, _ in ranked}
                    ranked = list(ranked)
                    for item in popular_items:
                        if item in blocked or item in selected_items:
                            continue
                        ranked.append((int(item), 0.0))
                        selected_items.add(int(item))
                        if len(ranked) >= out_k:
                            break
                score_by_item = dict(ranked[:out_k])
                normalized = normalize_scores(ranked[:out_k])
                for item, prob in normalized:
                    save_rows.append(
                        {
                            "user": user_id,
                            "item": int(item),
                            "entity_type": entity_type,
                            "probability": float(prob),
                            "candidate_score": float(score_by_item.get(item, 0.0)),
                        }
                    )

    summary = pd.DataFrame([stat.row() for stat in stats.values()])
    summary = summary.sort_values(["budget", "val_macro_recall", "strict_val_recall", "warmup_val_recall"], ascending=False)
    summary_path = args.output_dir / "summary.csv"
    summary.to_csv(summary_path, index=False)
    try:
        summary.to_excel(args.output_dir / "summary.xlsx", index=False)
    except Exception:
        pass

    meta = {
        "data_dir": str(data_dir),
        "seed": args.seed,
        "embedding_path": str(emb_path),
        "neighbor_ks": neighbor_ks,
        "policies": policies,
        "pool_m": args.pool_m,
        "strict_k": args.strict_k,
        "warmup_k": args.warmup_k,
        "device": device,
        "target_users": int(len(target_users)),
        "train_interactions": int(len(train_df)),
        "warm_source_users": int(len(warm_users_cpu)),
        "summary_csv": str(summary_path),
    }
    (args.output_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    if args.save_variant:
        save_csv = args.save_csv or (args.output_dir / f"{args.save_variant}.csv")
        save_df = pd.DataFrame(save_rows, columns=["user", "item", "entity_type", "probability", "candidate_score"])
        save_csv.parent.mkdir(parents=True, exist_ok=True)
        save_df.to_csv(save_csv, index=False)
        save_summary = {
            **meta,
            "save_variant": args.save_variant,
            "save_csv": str(save_csv),
            "selected_rows": int(len(save_df)),
            "selected_users": int(save_df["user"].nunique()) if len(save_df) else 0,
            "selected_items": int(save_df["item"].nunique()) if len(save_df) else 0,
            "entity_type_counts": save_df["entity_type"].value_counts().to_dict() if len(save_df) else {},
        }
        save_csv.with_suffix(".summary.json").write_text(
            json.dumps(save_summary, indent=2), encoding="utf-8"
        )

    print(summary.head(20).to_string(index=False))
    print(summary_path)


if __name__ == "__main__":
    main()
