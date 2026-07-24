import argparse
import os
import pickle
import random
from pprint import pprint

import numpy as np
import pandas as pd

import sys
sys.path.append("..")
import utils


def safe_read_csv(path, empty_ok=False):
    if os.path.exists(path):
        return pd.read_csv(path)
    if empty_ok:
        return pd.DataFrame(columns=["user", "item"])
    raise FileNotFoundError(f"Required file not found: {path}")


def unique_array(df, column):
    if column not in df.columns or len(df) == 0:
        return np.array([], dtype=np.int32)
    return np.array(sorted(set(df[column].tolist())), dtype=np.int32)


def complement_array(full_array, observed_array):
    return np.array(
        sorted(set(full_array.tolist()) - set(observed_array.tolist())),
        dtype=np.int32,
    )


def df_neighbors(df, key, size):
    if len(df) == 0:
        return np.array([np.array([], dtype=np.int32) for _ in range(size)], dtype=object)
    return utils.df_get_neighbors(df, key, size)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="CiteULike", help="Dataset to use.")
    parser.add_argument("--datadir", type=str, default="./", help="Directory of the dataset.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--cold_object", type=str, default="item", choices=["user", "item"])
    parser.add_argument("--protocol", type=str, default="warmup", choices=["original", "warmup"])
    parser.add_argument("--warmup_k", type=int, default=5, help="Stored in convert_dict for traceability.")
    args = parser.parse_args()
    pprint(vars(args))

    random.seed(args.seed)
    np.random.seed(args.seed)

    store_path = os.path.join(args.datadir, f"{args.dataset}/")
    procedure_timer = utils.Timer("Convert")

    df_train = safe_read_csv(os.path.join(store_path, "warm_emb.csv"))
    df_warm_train_original = safe_read_csv(os.path.join(store_path, "warm_emb_original.csv"), empty_ok=True)
    if len(df_warm_train_original) == 0:
        df_warm_train_original = df_train.copy()

    df_warm_val = safe_read_csv(os.path.join(store_path, "warm_val.csv"))
    df_warm_test = safe_read_csv(os.path.join(store_path, "warm_test.csv"))
    df_cold = safe_read_csv(os.path.join(store_path, f"cold_{args.cold_object}.csv"), empty_ok=True)
    df_cold_val = safe_read_csv(os.path.join(store_path, f"cold_{args.cold_object}_val.csv"))
    df_cold_test = safe_read_csv(os.path.join(store_path, f"cold_{args.cold_object}_test.csv"))

    df_warmup_support = safe_read_csv(os.path.join(store_path, "warmup_support.csv"), empty_ok=True)
    df_warmup_val = safe_read_csv(os.path.join(store_path, "warmup_val.csv"), empty_ok=True)
    df_warmup_test = safe_read_csv(os.path.join(store_path, "warmup_test.csv"), empty_ok=True)
    df_pos = safe_read_csv(os.path.join(store_path, f"{args.dataset}.csv"))

    if args.protocol == "warmup":
        df_overall_val = pd.concat([df_warm_val, df_warmup_val, df_cold_val], ignore_index=True)
        df_overall_test = pd.concat([df_warm_test, df_warmup_test, df_cold_test], ignore_index=True)
    else:
        warm_object = "user" if args.cold_object == "item" else "item"
        overall_val_object_set = set(df_cold_val[warm_object]) & set(df_warm_val[warm_object])
        overall_test_object_set = set(df_cold_test[warm_object]) & set(df_warm_test[warm_object])
        df_overall_val = pd.concat([df_cold_val, df_warm_val], ignore_index=True)
        df_overall_val = df_overall_val[df_overall_val[warm_object].isin(overall_val_object_set)]
        df_overall_test = pd.concat([df_cold_test, df_warm_test], ignore_index=True)
        df_overall_test = df_overall_test[df_overall_test[warm_object].isin(overall_test_object_set)]

    df_overall_val.to_csv(os.path.join(store_path, "overall_val.csv"), index=False)
    df_overall_test.to_csv(os.path.join(store_path, "overall_test.csv"), index=False)

    n_user_item = pickle.load(open(os.path.join(store_path, "n_user_item.pkl"), "rb"))
    user_num = n_user_item["user"]
    item_num = n_user_item["item"]
    procedure_timer.logging("Finish loading data.")
    print("Global user_num: {}  item_num: {}".format(user_num, item_num))
    print(
        "Split records: train={} warm_val={} warm_test={} warmup_val={} warmup_test={} strict_val={} strict_test={} overall_test={}".format(
            len(df_train),
            len(df_warm_val),
            len(df_warm_test),
            len(df_warmup_val),
            len(df_warmup_test),
            len(df_cold_val),
            len(df_cold_test),
            len(df_overall_test),
        )
    )

    user_array = np.arange(user_num, dtype=np.int32)
    item_array = np.arange(item_num, dtype=np.int32)

    df_warm_all = pd.concat([df_warm_train_original, df_warm_val, df_warm_test], ignore_index=True)
    df_warmup_all = pd.concat([df_warmup_support, df_warmup_val, df_warmup_test], ignore_index=True)
    df_strict_all = pd.concat([df_cold, df_cold_val, df_cold_test], ignore_index=True).drop_duplicates()

    warm_user = unique_array(df_warm_all, "user")
    warm_item = unique_array(df_warm_all, "item")
    warmup_user = unique_array(df_warmup_all, "user")
    warmup_item = unique_array(df_warmup_all, "item")
    strict_cold_user = unique_array(df_strict_all, "user")
    strict_cold_item = unique_array(df_strict_all, "item")
    if args.cold_object == "item":
        strict_cold_item = np.array(
            sorted(set(item_array.tolist()) - set(warm_item.tolist()) - set(warmup_item.tolist())),
            dtype=np.int32,
        )
    else:
        strict_cold_user = np.array(
            sorted(set(user_array.tolist()) - set(warm_user.tolist()) - set(warmup_user.tolist())),
            dtype=np.int32,
        )

    # Keep both the original warm graph and the graph after warm-up support
    # interactions are injected. In the pure graph, warm-up objects are cold;
    # in the mixed graph, they are visible training objects.
    pure_warm_user = warm_user
    pure_warm_item = warm_item
    pure_cold_user = complement_array(user_array, pure_warm_user)
    pure_cold_item = complement_array(item_array, pure_warm_item)

    mixed_warm_user = unique_array(df_train, "user")
    mixed_warm_item = unique_array(df_train, "item")
    mixed_cold_user = complement_array(user_array, mixed_warm_user)
    mixed_cold_item = complement_array(item_array, mixed_warm_item)

    emb_user_nb_mixed = df_neighbors(df_train, "user", user_num)
    emb_item_nb_mixed = df_neighbors(df_train, "item", item_num)
    emb_user_nb_pure = df_neighbors(df_warm_train_original, "user", user_num)
    emb_item_nb_pure = df_neighbors(df_warm_train_original, "item", item_num)

    para_dict = {
        "cold_object": args.cold_object,
        "protocol": args.protocol,
        "warmup_k": args.warmup_k,
        "user_num": user_num,
        "item_num": item_num,
        "user_array": user_array,
        "item_array": item_array,
        "emb_user": mixed_warm_user,
        "emb_item": mixed_warm_item,
        "emb_user_pure": unique_array(df_warm_train_original, "user"),
        "emb_item_pure": unique_array(df_warm_train_original, "item"),
        "train_user": mixed_warm_user,
        "train_item": mixed_warm_item,
        "warm_user": warm_user,
        "warm_item": warm_item,
        "warmup_user": warmup_user,
        "warmup_item": warmup_item,
        "cold_user": strict_cold_user,
        "cold_item": strict_cold_item,
        "strict_cold_user": strict_cold_user,
        "strict_cold_item": strict_cold_item,
        "pure_warm_user": pure_warm_user,
        "pure_warm_item": pure_warm_item,
        "pure_cold_user": pure_cold_user,
        "pure_cold_item": pure_cold_item,
        "mixed_warm_user": mixed_warm_user,
        "mixed_warm_item": mixed_warm_item,
        "mixed_cold_user": mixed_cold_user,
        "mixed_cold_item": mixed_cold_item,
        "warm_val_user": unique_array(df_warm_val, "user"),
        "warm_test_user": unique_array(df_warm_test, "user"),
        "warmup_val_user": unique_array(df_warmup_val, "user"),
        "warmup_test_user": unique_array(df_warmup_test, "user"),
        "cold_val_user": unique_array(df_cold_val, "user"),
        "cold_test_user": unique_array(df_cold_test, "user"),
        "overall_val_user": unique_array(df_overall_val, "user"),
        "overall_test_user": unique_array(df_overall_test, "user"),
        "warm_val_item": unique_array(df_warm_val, "item"),
        "warm_test_item": unique_array(df_warm_test, "item"),
        "warmup_val_item": unique_array(df_warmup_val, "item"),
        "warmup_test_item": unique_array(df_warmup_test, "item"),
        "cold_val_item": unique_array(df_cold_val, "item"),
        "cold_test_item": unique_array(df_cold_test, "item"),
        "overall_val_item": unique_array(df_overall_val, "item"),
        "overall_test_item": unique_array(df_overall_test, "item"),
        "pos_user_nb": df_neighbors(df_pos, "user", user_num),
        "emb_user_nb": emb_user_nb_mixed,
        "emb_user_nb_mixed": emb_user_nb_mixed,
        "emb_user_nb_pure": emb_user_nb_pure,
        "warm_val_user_nb": df_neighbors(df_warm_val, "user", user_num),
        "warm_test_user_nb": df_neighbors(df_warm_test, "user", user_num),
        "warmup_val_user_nb": df_neighbors(df_warmup_val, "user", user_num),
        "warmup_test_user_nb": df_neighbors(df_warmup_test, "user", user_num),
        "cold_val_user_nb": df_neighbors(df_cold_val, "user", user_num),
        "cold_test_user_nb": df_neighbors(df_cold_test, "user", user_num),
        "overall_val_user_nb": df_neighbors(df_overall_val, "user", user_num),
        "overall_test_user_nb": df_neighbors(df_overall_test, "user", user_num),
        "emb_item_nb": emb_item_nb_mixed,
        "emb_item_nb_mixed": emb_item_nb_mixed,
        "emb_item_nb_pure": emb_item_nb_pure,
    }

    dict_path = os.path.join(store_path, "convert_dict.pkl")
    pickle.dump(para_dict, open(dict_path, "wb"), protocol=4)
    procedure_timer.logging("Convert {} successfully, store the dict to {}".format(args.dataset, dict_path))

    print("[Entity groups]")
    print("warm user/item: {} / {}".format(len(para_dict["warm_user"]), len(para_dict["warm_item"])))
    print("warmup user/item: {} / {}".format(len(para_dict["warmup_user"]), len(para_dict["warmup_item"])))
    print("strict cold user/item: {} / {}".format(len(para_dict["cold_user"]), len(para_dict["cold_item"])))
    print("pure warm user/item: {} / {}".format(len(para_dict["pure_warm_user"]), len(para_dict["pure_warm_item"])))
    print("pure cold user/item: {} / {}".format(len(para_dict["pure_cold_user"]), len(para_dict["pure_cold_item"])))
    print("mixed warm user/item: {} / {}".format(len(para_dict["mixed_warm_user"]), len(para_dict["mixed_warm_item"])))
    print("mixed cold user/item: {} / {}".format(len(para_dict["mixed_cold_user"]), len(para_dict["mixed_cold_item"])))


if __name__ == "__main__":
    main()
