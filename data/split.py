import argparse
import ast
import json
import os
import pickle
import random
from pprint import pprint

import numpy as np
import pandas as pd


def parse_ratio_list(value):
    parsed = ast.literal_eval(value)
    if not isinstance(parsed, (list, tuple)):
        raise ValueError(f"Expected a list/tuple ratio, got: {value}")
    return [float(item) for item in parsed]


def get_group_list(dataframe, by_key):
    return [(entity_id, group.index.to_numpy(dtype=np.int64)) for entity_id, group in dataframe.groupby(by=by_key)]


def safe_concat(index_groups):
    if not index_groups:
        return np.array([], dtype=np.int64)
    return np.concatenate(index_groups, axis=0).astype(np.int64)


def split_train_val_test(df_all, idx, warm_split):
    n_val = int(warm_split[1] * len(idx))
    n_test = int(warm_split[2] * len(idx))
    n_train = len(idx) - n_val - n_test

    idx = idx.copy()
    np.random.shuffle(idx)
    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train + n_val]
    test_idx = idx[-n_test:] if n_test > 0 else np.array([], dtype=np.int64)
    original_train_len = len(train_idx)

    # Keep validation/test users and items visible in the training graph.
    for split_name, current_idx in (("val", val_idx), ("test", test_idx)):
        original_len = len(current_idx)
        for column in ("user", "item"):
            train_set = set(df_all.loc[train_idx, column])
            split_df = df_all.loc[current_idx]
            idx_to_move = split_df[~split_df[column].isin(train_set)].index.to_numpy(dtype=np.int64)
            if len(idx_to_move) > 0:
                current_idx = np.array(
                    sorted(set(current_idx.tolist()) - set(idx_to_move.tolist())),
                    dtype=np.int64,
                )
                train_idx = np.concatenate([train_idx, idx_to_move], axis=0)
        if split_name == "val":
            val_idx = current_idx
        else:
            test_idx = current_idx
        print(f"Warm {split_name} splitting finished: {original_len} -> {len(current_idx)}")

    print(f"Warm train splitting finished: {original_train_len} -> {len(train_idx)}")
    return train_idx.astype(np.int64), val_idx.astype(np.int64), test_idx.astype(np.int64)


def filter_by_visible_opposite(df_all, idx, warm_object, visible_objects):
    if len(idx) == 0:
        return idx
    split_df = df_all.loc[idx]
    filtered_idx = split_df[split_df[warm_object].isin(visible_objects)].index.to_numpy(dtype=np.int64)
    if len(filtered_idx) != len(idx):
        print(f"Filter {warm_object}-visible eval interactions: {len(idx)} -> {len(filtered_idx)}")
    return filtered_idx


def split_strict_cold(df_all, strict_cold_idx, cold_object, cold_split, warm_object, visible_objects):
    strict_cold_idx = filter_by_visible_opposite(df_all, strict_cold_idx, warm_object, visible_objects)
    if len(strict_cold_idx) == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64), np.array([], dtype=np.int64)

    df_cold = df_all.loc[strict_cold_idx]
    cold_group = np.array([group.index.to_numpy(dtype=np.int64) for _, group in df_cold.groupby(by=cold_object)], dtype=object)
    np.random.shuffle(cold_group)

    if len(cold_group) <= 1:
        cold_val_idx = np.array([], dtype=np.int64)
        cold_test_idx = safe_concat(cold_group.tolist())
    else:
        n_val_group = int(cold_split[0] * len(cold_group))
        n_val_group = min(max(n_val_group, 1), len(cold_group) - 1)
        cold_val_idx = safe_concat(cold_group[:n_val_group].tolist())
        cold_test_idx = safe_concat(cold_group[n_val_group:].tolist())

    strict_cold_eval_idx = safe_concat([cold_val_idx, cold_test_idx])
    return strict_cold_eval_idx, cold_val_idx, cold_test_idx


def sample_warmup_splits(df_all, warmup_group, k0):
    support_idx = []
    val_idx = []
    test_idx = []
    rng = np.random.RandomState(np.random.randint(0, 2**31 - 1))

    for entity_id, entity_idx in warmup_group:
        entity_idx = entity_idx.copy()
        rng.shuffle(entity_idx)

        if len(entity_idx) < k0 + 1:
            print(f"[Warning] Drop warm-up entity {entity_id}: only {len(entity_idx)} interactions.")
            continue

        support_k = rng.randint(1, k0)
        current_support = entity_idx[:support_k]
        rest_idx = entity_idx[support_k:]
        n_val = len(rest_idx) // 2

        current_val = rest_idx[:n_val]
        current_test = rest_idx[n_val:]
        if len(current_val) == 0 or len(current_test) == 0:
            print(f"[Warning] Drop warm-up entity {entity_id}: empty val/test after support sampling.")
            continue

        support_idx.extend(current_support.tolist())
        val_idx.extend(current_val.tolist())
        test_idx.extend(current_test.tolist())

    return (
        np.array(support_idx, dtype=np.int64),
        np.array(val_idx, dtype=np.int64),
        np.array(test_idx, dtype=np.int64),
    )


def count_stats(df):
    if len(df) == 0:
        return {"users": 0, "items": 0, "records": 0}
    return {
        "users": int(df["user"].nunique()),
        "items": int(df["item"].nunique()),
        "records": int(len(df)),
    }


def write_split(df, path):
    df[["user", "item"]].to_csv(path, index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="CiteULike", help="Dataset to use.")
    parser.add_argument("--datadir", type=str, default="./", help="Directory of the dataset.")
    parser.add_argument("--warm_ratio", type=float, default=0.8, help="Warm entity ratio.")
    parser.add_argument("--cold_ratio", type=float, default=0.1, help="Strict cold-start entity ratio.")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warm-up entity ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--warm_split", default="[0.8, 0.1, 0.1]", help="Warm train/val/test split.")
    parser.add_argument("--cold_split", default="[0.5, 0.5]", help="Strict cold val/test split.")
    parser.add_argument("--warmup_k", type=int, default=5, help="Warm-up threshold k0; support is sampled from 1..k0-1.")
    parser.add_argument("--cold_object", type=str, default="item", choices=["user", "item"])
    args = parser.parse_args()
    args.warm_split = parse_ratio_list(args.warm_split)
    args.cold_split = parse_ratio_list(args.cold_split)
    pprint(vars(args))

    if abs(args.warm_ratio + args.cold_ratio + args.warmup_ratio - 1.0) > 1e-8:
        raise ValueError("warm_ratio + cold_ratio + warmup_ratio must sum to 1.0")
    if args.warmup_k <= 1:
        raise ValueError("warmup_k is the threshold k0 and must be greater than 1")

    random.seed(args.seed)
    np.random.seed(args.seed)

    store_path = os.path.join(args.datadir, args.dataset)
    if not os.path.exists(store_path):
        raise FileNotFoundError(f"Store path {store_path} not found!")

    df = pd.read_csv(
        os.path.join(store_path, args.dataset + ".csv"),
        header=0,
        usecols=["user", "item"],
        index_col=False,
        dtype={"user": np.int64, "item": np.int64},
    )
    origin_len = df.shape[0]
    df = df.drop_duplicates(["user", "item"]).reset_index(drop=True)
    print("Duplicated :%d -> %d" % (origin_len, df.shape[0]))

    user_num = int(max(df["user"]) + 1)
    item_num = int(max(df["item"]) + 1)
    with open(os.path.join(store_path, "n_user_item.pkl"), "wb") as f:
        pickle.dump({"user": user_num, "item": item_num}, f)
    print("User: %d\tItem: %d" % (user_num, item_num))
    print("Global sparse rate: %.4f" % ((user_num * item_num - len(df)) / (user_num * item_num) * 100.0))

    group = get_group_list(df, args.cold_object)
    random.shuffle(group)
    n_total_group = len(group)
    n_warm_group = int(args.warm_ratio * n_total_group)
    n_cold_group = int(args.cold_ratio * n_total_group)
    n_warmup_group_target = n_total_group - n_warm_group - n_cold_group

    warm_group = group[:n_warm_group]
    remaining_group = group[n_warm_group:]
    min_warmup_interactions = args.warmup_k + 1
    eligible_warmup_group = [g for g in remaining_group if len(g[1]) >= min_warmup_interactions]
    ineligible_warmup_group = [g for g in remaining_group if len(g[1]) < min_warmup_interactions]
    warmup_group = eligible_warmup_group[:n_warmup_group_target]
    strict_cold_group = eligible_warmup_group[n_warmup_group_target:] + ineligible_warmup_group

    if len(warmup_group) < n_warmup_group_target:
        print(
            f"[Warning] Only {len(warmup_group)} warm-up entities are feasible; "
            f"target was {n_warmup_group_target}. Minimum full-history interactions: {min_warmup_interactions}."
        )

    warm_idx = safe_concat([idx for _, idx in warm_group])
    warmup_idx = safe_concat([idx for _, idx in warmup_group])
    strict_cold_idx = safe_concat([idx for _, idx in strict_cold_group])

    print("[Entity Split]\tentities\trecords")
    print(f"warm\t{len(warm_group)}\t{len(warm_idx)}")
    print(f"strict_cold\t{len(strict_cold_group)}\t{len(strict_cold_idx)}")
    print(f"warmup\t{len(warmup_group)}\t{len(warmup_idx)}")

    warm_train_original_idx, warm_val_idx, warm_test_idx = split_train_val_test(df, warm_idx, args.warm_split)
    warmup_support_idx, warmup_val_idx, warmup_test_idx = sample_warmup_splits(df, warmup_group, args.warmup_k)

    final_train_idx = warm_train_original_idx.copy()
    if len(warmup_support_idx) > 0:
        final_train_idx = np.concatenate([final_train_idx, warmup_support_idx], axis=0)

    warm_object = "user" if args.cold_object == "item" else "item"
    visible_objects = set(df.loc[final_train_idx, warm_object])
    warmup_val_idx = filter_by_visible_opposite(df, warmup_val_idx, warm_object, visible_objects)
    warmup_test_idx = filter_by_visible_opposite(df, warmup_test_idx, warm_object, visible_objects)
    strict_cold_eval_idx, cold_val_idx, cold_test_idx = split_strict_cold(
        df, strict_cold_idx, args.cold_object, args.cold_split, warm_object, visible_objects
    )

    df_warm_train_original = df.loc[warm_train_original_idx]
    df_train = df.loc[final_train_idx]
    df_warm_val = df.loc[warm_val_idx]
    df_warm_test = df.loc[warm_test_idx]
    df_warmup_support = df.loc[warmup_support_idx] if len(warmup_support_idx) > 0 else df.iloc[0:0].copy()
    df_warmup_val = df.loc[warmup_val_idx]
    df_warmup_test = df.loc[warmup_test_idx]
    df_cold = df.loc[strict_cold_eval_idx]
    df_cold_val = df.loc[cold_val_idx]
    df_cold_test = df.loc[cold_test_idx]
    df_overall_val = pd.concat([df_warm_val, df_warmup_val, df_cold_val], ignore_index=True)
    df_overall_test = pd.concat([df_warm_test, df_warmup_test, df_cold_test], ignore_index=True)

    write_split(df_train, os.path.join(store_path, "warm_emb.csv"))
    write_split(df_train, os.path.join(store_path, "warm_train.csv"))
    write_split(df_warm_train_original, os.path.join(store_path, "warm_emb_original.csv"))
    write_split(df_warm_train_original, os.path.join(store_path, "warm_train_original.csv"))
    write_split(df_warm_val, os.path.join(store_path, "warm_val.csv"))
    write_split(df_warm_test, os.path.join(store_path, "warm_test.csv"))
    write_split(df_warmup_support, os.path.join(store_path, "warmup_support.csv"))
    write_split(df_warmup_val, os.path.join(store_path, "warmup_val.csv"))
    write_split(df_warmup_test, os.path.join(store_path, "warmup_test.csv"))
    write_split(df_cold, os.path.join(store_path, f"cold_{args.cold_object}.csv"))
    write_split(df_cold_val, os.path.join(store_path, f"cold_{args.cold_object}_val.csv"))
    write_split(df_cold_test, os.path.join(store_path, f"cold_{args.cold_object}_test.csv"))
    write_split(df_overall_val, os.path.join(store_path, "overall_val.csv"))
    write_split(df_overall_test, os.path.join(store_path, "overall_test.csv"))
    df.to_csv(os.path.join(store_path, args.dataset + ".csv"), index=False)

    # Backward-compatible explicit aliases for item-side strict cold-start.
    if args.cold_object == "item":
        write_split(df_cold_val, os.path.join(store_path, "strict_cold_item_val.csv"))
        write_split(df_cold_test, os.path.join(store_path, "strict_cold_item_test.csv"))

    summary = {
        "seed": args.seed,
        "cold_object": args.cold_object,
        "warmup_k": args.warmup_k,
        "warm": count_stats(pd.concat([df_warm_train_original, df_warm_val, df_warm_test], ignore_index=True)),
        "train": count_stats(df_train),
        "warmup_support": count_stats(df_warmup_support),
        "warmup_val": count_stats(df_warmup_val),
        "warmup_test": count_stats(df_warmup_test),
        "strict_cold_val": count_stats(df_cold_val),
        "strict_cold_test": count_stats(df_cold_test),
        "overall_val": count_stats(df_overall_val),
        "overall_test": count_stats(df_overall_test),
    }
    with open(os.path.join(store_path, "split_meta.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("[Stored Split Summary]")
    for name, stats in summary.items():
        if isinstance(stats, dict):
            print(f"{name}\tusers={stats['users']}\titems={stats['items']}\trecords={stats['records']}")
    print("Split finished!")


if __name__ == "__main__":
    main()
