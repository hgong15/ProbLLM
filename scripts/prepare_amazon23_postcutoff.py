#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import pickle
import random
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from huggingface_hub import HfFileSystem
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


HF_REPO = "McAuley-Lab/Amazon-Reviews-2023"
DATE_FMT = "%Y-%m-%d"


def parse_date(value: str) -> datetime:
    return datetime.strptime(value, DATE_FMT).replace(tzinfo=timezone.utc)


def timestamp_to_dt(value) -> datetime:
    ts = int(value)
    if ts > 10_000_000_000:
        ts = ts / 1000.0
    return datetime.fromtimestamp(ts, tz=timezone.utc)


def is_verified(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}


def jsonl_remote_path(kind: str, category: str) -> str:
    if kind == "review":
        return f"datasets/{HF_REPO}/raw/review_categories/{category}.jsonl"
    if kind == "meta":
        return f"datasets/{HF_REPO}/raw/meta_categories/meta_{category}.jsonl"
    raise ValueError(kind)


def iter_jsonl(fs: HfFileSystem, path: str) -> Iterable[dict]:
    with fs.open(path, "r") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Could not parse {path}:{line_no}") from exc


def collect_reviews(args) -> pd.DataFrame:
    fs = HfFileSystem()
    start_dt = parse_date(args.start_date)
    end_dt = parse_date(args.end_date)
    rows: list[dict] = []
    seen = 0
    kept = 0
    review_path = jsonl_remote_path("review", args.category)
    for obj in iter_jsonl(fs, review_path):
        seen += 1
        if seen % args.progress_every == 0:
            print(f"[reviews] scanned={seen:,} kept={kept:,}", flush=True)
        try:
            rating = float(obj.get("rating", 0.0))
            dt = timestamp_to_dt(obj["timestamp"])
        except Exception:
            continue
        if dt < start_dt or dt > end_dt:
            continue
        if rating < args.min_rating:
            continue
        if args.verified_only and not is_verified(obj.get("verified_purchase")):
            continue
        user_id = str(obj.get("user_id", "")).strip()
        parent_asin = str(obj.get("parent_asin") or obj.get("asin") or "").strip()
        if not user_id or not parent_asin:
            continue
        rows.append(
            {
                "raw_user": user_id,
                "raw_item": parent_asin,
                "rating": rating,
                "timestamp": int(obj["timestamp"]),
                "datetime": dt.isoformat(),
                "review_title": str(obj.get("title", ""))[:300],
            }
        )
        kept += 1
        if args.max_raw_reviews and kept >= args.max_raw_reviews:
            break
    if not rows:
        raise RuntimeError(f"No post-cutoff positive reviews found for {args.category}")
    df = pd.DataFrame(rows)
    df = df.sort_values(["timestamp", "raw_user", "raw_item"])
    before = len(df)
    df = df.drop_duplicates(["raw_user", "raw_item"], keep="first").reset_index(drop=True)
    print(f"[reviews] category={args.category} scanned={seen:,} kept={kept:,} dedup={before:,}->{len(df):,}", flush=True)
    return df


def stringify(value, max_items: int = 8) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, dict):
        parts = []
        for key, val in list(value.items())[:max_items]:
            parts.append(f"{key}: {stringify(val, max_items=4)}")
        return "; ".join(parts)
    if isinstance(value, (list, tuple)):
        return "; ".join(stringify(v, max_items=4) for v in list(value)[:max_items])
    return str(value)


def collect_metadata(args, needed_items: set[str]) -> pd.DataFrame:
    fs = HfFileSystem()
    rows: list[dict] = []
    seen = 0
    meta_path = jsonl_remote_path("meta", args.category)
    for obj in iter_jsonl(fs, meta_path):
        seen += 1
        if seen % args.progress_every == 0:
            print(f"[meta] scanned={seen:,} matched={len(rows):,}/{len(needed_items):,}", flush=True)
        parent_asin = str(obj.get("parent_asin") or obj.get("asin") or "").strip()
        if parent_asin not in needed_items:
            continue
        title = stringify(obj.get("title"))
        categories = stringify(obj.get("categories"))
        features = stringify(obj.get("features"))
        description = stringify(obj.get("description"))
        details = stringify(obj.get("details"))
        text = " ".join(part for part in [title, categories, features, description, details] if part)
        if not title and not text:
            continue
        rows.append(
            {
                "raw_item": parent_asin,
                "title": title,
                "categories": categories,
                "features": features,
                "description": description,
                "details": details,
                "average_rating": obj.get("average_rating", ""),
                "rating_number": obj.get("rating_number", ""),
                "metadata_text": text[:8000],
            }
        )
        if len(rows) >= len(needed_items):
            break
    if not rows:
        raise RuntimeError(f"No metadata matched for {args.category}")
    meta = pd.DataFrame(rows).drop_duplicates(["raw_item"], keep="first")
    print(f"[meta] category={args.category} scanned={seen:,} matched={len(meta):,}/{len(needed_items):,}", flush=True)
    return meta


def iterative_filter(df: pd.DataFrame, min_user_count: int, min_item_count: int) -> pd.DataFrame:
    changed = True
    out = df.copy()
    while changed:
        before = len(out)
        if min_user_count > 1:
            user_counts = out["raw_user"].value_counts()
            out = out[out["raw_user"].isin(user_counts[user_counts >= min_user_count].index)]
        if min_item_count > 1:
            item_counts = out["raw_item"].value_counts()
            out = out[out["raw_item"].isin(item_counts[item_counts >= min_item_count].index)]
        changed = len(out) != before
    return out.reset_index(drop=True)


def sample_eval_entities(df: pd.DataFrame, cold_object: str, max_eval_entities: int, seed: int) -> pd.DataFrame:
    if max_eval_entities <= 0:
        return df
    rng = random.Random(seed)
    train_df = df[df["period"] == "train"]
    eval_df = df[df["period"].isin(["val", "test"])]
    train_counts = train_df[cold_object].value_counts()
    eval_entities = sorted(eval_df[cold_object].unique().tolist())
    if len(eval_entities) <= max_eval_entities:
        return df
    strict = [e for e in eval_entities if int(train_counts.get(e, 0)) == 0]
    warmup = [e for e in eval_entities if 0 < int(train_counts.get(e, 0)) < 5]
    warm = [e for e in eval_entities if int(train_counts.get(e, 0)) >= 5]
    sampled: list[int] = []
    for bucket in (strict, warmup, warm):
        rng.shuffle(bucket)
    quotas = [math.ceil(max_eval_entities * 0.4), math.ceil(max_eval_entities * 0.3), max_eval_entities]
    sampled.extend(strict[: quotas[0]])
    sampled.extend(warmup[: quotas[1]])
    remaining = max_eval_entities - len(set(sampled))
    if remaining > 0:
        sampled.extend(warm[:remaining])
    remaining = max_eval_entities - len(set(sampled))
    if remaining > 0:
        rest = [e for e in eval_entities if e not in set(sampled)]
        rng.shuffle(rest)
        sampled.extend(rest[:remaining])
    sampled_set = set(sampled[:max_eval_entities])
    train_entities = set(train_df[cold_object].unique().tolist())
    keep_train = train_df[train_df[cold_object].isin(train_entities)]
    keep_eval = eval_df[eval_df[cold_object].isin(sampled_set)]
    return pd.concat([keep_train, keep_eval], ignore_index=True).drop_duplicates(["user", "item", "period"])


def write_pairs(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df[["user", "item"]].drop_duplicates().to_csv(path, index=False)


def unique_array(df: pd.DataFrame, column: str) -> np.ndarray:
    if len(df) == 0:
        return np.asarray([], dtype=np.int32)
    return np.asarray(sorted(df[column].unique().tolist()), dtype=np.int32)


def df_neighbors(df: pd.DataFrame, key: str, size: int) -> np.ndarray:
    grouped: dict[int, list[int]] = defaultdict(list)
    other = "item" if key == "user" else "user"
    for key_value, other_value in df[[key, other]].itertuples(index=False):
        grouped[int(key_value)].append(int(other_value))
    out = np.empty(size, dtype=object)
    for idx in range(size):
        out[idx] = np.asarray(sorted(set(grouped.get(idx, []))), dtype=np.int32)
    return out


def count_stats(df: pd.DataFrame) -> dict[str, int]:
    return {
        "users": int(df["user"].nunique()) if len(df) else 0,
        "items": int(df["item"].nunique()) if len(df) else 0,
        "records": int(len(df)),
    }


def build_content(meta: pd.DataFrame, item_num: int, out_path: Path, dim: int) -> None:
    texts = meta.sort_values("item")["metadata_text"].fillna("").astype(str).tolist()
    vectorizer = TfidfVectorizer(
        max_features=max(512, min(16384, item_num * 8)),
        min_df=1 if item_num < 1000 else 2,
        stop_words="english",
        ngram_range=(1, 2),
        dtype=np.float32,
    )
    tfidf = vectorizer.fit_transform(texts)
    n_comp = min(dim, max(2, min(tfidf.shape[0] - 1, tfidf.shape[1] - 1)))
    if n_comp < 2:
        arr = tfidf.toarray().astype(np.float32)
        if arr.shape[1] < dim:
            arr = np.pad(arr, ((0, 0), (0, dim - arr.shape[1])), mode="constant")
        arr = arr[:, :dim]
    else:
        svd = TruncatedSVD(n_components=n_comp, random_state=42)
        arr = svd.fit_transform(tfidf).astype(np.float32)
        if arr.shape[1] < dim:
            arr = np.pad(arr, ((0, 0), (0, dim - arr.shape[1])), mode="constant")
    arr = normalize(arr, norm="l2", axis=1).astype(np.float32)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, arr)


def split_temporal(df: pd.DataFrame, args) -> dict[str, pd.DataFrame | dict]:
    out = df.copy()
    dt = pd.to_datetime(out["datetime"], utc=True, format="mixed")
    if args.split_strategy == "relative":
        out = out.assign(_dt=dt).sort_values(["_dt", "raw_user", "raw_item"]).reset_index(drop=True)
        n_rows = len(out)
        train_end = max(1, min(n_rows - 2, int(math.floor(n_rows * args.relative_train_frac))))
        val_size = max(1, int(math.floor(n_rows * args.relative_val_frac)))
        val_end = max(train_end + 1, min(n_rows - 1, train_end + val_size))
        out["period"] = "test"
        out.loc[: train_end - 1, "period"] = "train"
        out.loc[train_end : val_end - 1, "period"] = "val"
        if args.ensure_visible_opposite:
            warm_object = "user" if args.cold_object == "item" else "item"
            train_seen = set(out.loc[out["period"].eq("train"), warm_object].tolist())
            late = out.loc[~out["period"].eq("train"), [warm_object, "_dt"]]
            anchors = []
            for value, group in late.groupby(warm_object, sort=False):
                if value in train_seen:
                    continue
                anchors.append(group.sort_values("_dt").index[0])
            if anchors:
                out.loc[anchors, "period"] = "train"
        out = out.drop(columns=["_dt"])
    else:
        train_start = parse_date(args.train_start)
        val_start = parse_date(args.val_start)
        test_start = parse_date(args.test_start)
        end_dt = parse_date(args.end_date)
        out["period"] = np.where(
            (dt >= train_start) & (dt < val_start),
            "train",
            np.where((dt >= val_start) & (dt < test_start), "val", np.where((dt >= test_start) & (dt <= end_dt), "test", "drop")),
        )
        out = out[out["period"] != "drop"].copy()
    train_df = out[out["period"] == "train"].copy()
    val_df = out[out["period"] == "val"].copy()
    test_df = out[out["period"] == "test"].copy()
    if len(train_df) == 0 or len(test_df) == 0:
        raise RuntimeError(f"Temporal split is empty: train={len(train_df)} test={len(test_df)}")

    cold_object = args.cold_object
    warm_object = "user" if cold_object == "item" else "item"
    train_counts = train_df[cold_object].value_counts()
    visible_opposite = set(train_df[warm_object].unique().tolist())

    def filter_eval(split: pd.DataFrame) -> pd.DataFrame:
        return split[split[warm_object].isin(visible_opposite)].copy()

    val_df = filter_eval(val_df)
    test_df = filter_eval(test_df)
    eval_entities = set(pd.concat([val_df, test_df], ignore_index=True)[cold_object].unique().tolist())
    strict_entities = sorted([e for e in eval_entities if int(train_counts.get(e, 0)) == 0])
    warmup_entities = sorted([e for e in eval_entities if 0 < int(train_counts.get(e, 0)) < args.warmup_k])
    warm_entities = sorted([e for e in eval_entities if int(train_counts.get(e, 0)) >= args.warmup_k])

    def by_entities(split: pd.DataFrame, entities: list[int]) -> pd.DataFrame:
        if not entities:
            return split.iloc[0:0].copy()
        return split[split[cold_object].isin(set(entities))].copy()

    warm_train_original = train_df[train_df[cold_object].isin(set(warm_entities))].copy()
    warmup_support = train_df[train_df[cold_object].isin(set(warmup_entities))].copy()
    final_train = pd.concat([warm_train_original, warmup_support], ignore_index=True).drop_duplicates(["user", "item"])
    strict_val = by_entities(val_df, strict_entities)
    strict_test = by_entities(test_df, strict_entities)
    warmup_val = by_entities(val_df, warmup_entities)
    warmup_test = by_entities(test_df, warmup_entities)
    warm_val = by_entities(val_df, warm_entities)
    warm_test = by_entities(test_df, warm_entities)
    overall_val = pd.concat([strict_val, warmup_val, warm_val], ignore_index=True).drop_duplicates(["user", "item"])
    overall_test = pd.concat([strict_test, warmup_test, warm_test], ignore_index=True).drop_duplicates(["user", "item"])
    strict_all = pd.concat([strict_val, strict_test], ignore_index=True).drop_duplicates(["user", "item"])

    return {
        "full": out,
        "train": final_train,
        "warm_train_original": warm_train_original,
        "warmup_support": warmup_support,
        "warm_val": warm_val,
        "warm_test": warm_test,
        "warmup_val": warmup_val,
        "warmup_test": warmup_test,
        f"cold_{cold_object}": strict_all,
        f"cold_{cold_object}_val": strict_val,
        f"cold_{cold_object}_test": strict_test,
        "overall_val": overall_val,
        "overall_test": overall_test,
        "entity_groups": {
            "strict": strict_entities,
            "warmup": warmup_entities,
            "warm": warm_entities,
        },
    }


def write_convert_dict(out_dir: Path, dataset: str, cold_object: str, splits: dict, user_num: int, item_num: int, warmup_k: int) -> dict:
    df_full = splits["full"]
    df_train = splits["train"]
    df_warm_train_original = splits["warm_train_original"]
    df_warm_val = splits["warm_val"]
    df_warm_test = splits["warm_test"]
    df_warmup_support = splits["warmup_support"]
    df_warmup_val = splits["warmup_val"]
    df_warmup_test = splits["warmup_test"]
    df_cold = splits[f"cold_{cold_object}"]
    df_cold_val = splits[f"cold_{cold_object}_val"]
    df_cold_test = splits[f"cold_{cold_object}_test"]
    df_overall_val = splits["overall_val"]
    df_overall_test = splits["overall_test"]

    user_array = np.arange(user_num, dtype=np.int32)
    item_array = np.arange(item_num, dtype=np.int32)
    df_warm_all = pd.concat([df_warm_train_original, df_warm_val, df_warm_test], ignore_index=True)
    df_warmup_all = pd.concat([df_warmup_support, df_warmup_val, df_warmup_test], ignore_index=True)
    df_strict_all = pd.concat([df_cold, df_cold_val, df_cold_test], ignore_index=True).drop_duplicates(["user", "item"])

    warm_user = unique_array(df_warm_all, "user")
    warm_item = unique_array(df_warm_all, "item")
    warmup_user = unique_array(df_warmup_all, "user")
    warmup_item = unique_array(df_warmup_all, "item")
    strict_cold_user = unique_array(df_strict_all, "user")
    strict_cold_item = unique_array(df_strict_all, "item")
    if cold_object == "item":
        strict_cold_item = np.asarray(sorted(splits["entity_groups"]["strict"]), dtype=np.int32)
    else:
        strict_cold_user = np.asarray(sorted(splits["entity_groups"]["strict"]), dtype=np.int32)

    para = {
        "cold_object": cold_object,
        "protocol": "amazon23_postcutoff_temporal",
        "warmup_k": int(warmup_k),
        "user_num": int(user_num),
        "item_num": int(item_num),
        "user_array": user_array,
        "item_array": item_array,
        "emb_user": unique_array(df_train, "user"),
        "emb_item": unique_array(df_train, "item"),
        "emb_user_pure": unique_array(df_warm_train_original, "user"),
        "emb_item_pure": unique_array(df_warm_train_original, "item"),
        "train_user": unique_array(df_train, "user"),
        "train_item": unique_array(df_train, "item"),
        "warm_user": warm_user,
        "warm_item": warm_item,
        "warmup_user": warmup_user,
        "warmup_item": warmup_item,
        "cold_user": strict_cold_user,
        "cold_item": strict_cold_item,
        "strict_cold_user": strict_cold_user,
        "strict_cold_item": strict_cold_item,
        "pure_warm_user": warm_user,
        "pure_warm_item": warm_item,
        "pure_cold_user": np.asarray(sorted(set(user_array.tolist()) - set(warm_user.tolist())), dtype=np.int32),
        "pure_cold_item": np.asarray(sorted(set(item_array.tolist()) - set(warm_item.tolist())), dtype=np.int32),
        "mixed_warm_user": unique_array(df_train, "user"),
        "mixed_warm_item": unique_array(df_train, "item"),
        "mixed_cold_user": np.asarray(sorted(set(user_array.tolist()) - set(unique_array(df_train, "user").tolist())), dtype=np.int32),
        "mixed_cold_item": np.asarray(sorted(set(item_array.tolist()) - set(unique_array(df_train, "item").tolist())), dtype=np.int32),
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
        "pos_user_nb": df_neighbors(df_full, "user", user_num),
        "emb_user_nb": df_neighbors(df_train, "user", user_num),
        "emb_user_nb_mixed": df_neighbors(df_train, "user", user_num),
        "emb_user_nb_pure": df_neighbors(df_warm_train_original, "user", user_num),
        "warm_val_user_nb": df_neighbors(df_warm_val, "user", user_num),
        "warm_test_user_nb": df_neighbors(df_warm_test, "user", user_num),
        "warmup_val_user_nb": df_neighbors(df_warmup_val, "user", user_num),
        "warmup_test_user_nb": df_neighbors(df_warmup_test, "user", user_num),
        "cold_val_user_nb": df_neighbors(df_cold_val, "user", user_num),
        "cold_test_user_nb": df_neighbors(df_cold_test, "user", user_num),
        "overall_val_user_nb": df_neighbors(df_overall_val, "user", user_num),
        "overall_test_user_nb": df_neighbors(df_overall_test, "user", user_num),
        "emb_item_nb": df_neighbors(df_train, "item", item_num),
        "emb_item_nb_mixed": df_neighbors(df_train, "item", item_num),
        "emb_item_nb_pure": df_neighbors(df_warm_train_original, "item", item_num),
    }
    pickle.dump(para, open(out_dir / "convert_dict.pkl", "wb"), protocol=4)
    pickle.dump({"user": int(user_num), "item": int(item_num)}, open(out_dir / "n_user_item.pkl", "wb"), protocol=4)
    return para


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare post-cutoff Amazon Reviews'23 temporal cold-start data.")
    parser.add_argument("--category", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--cold_object", choices=["item"], default="item")
    parser.add_argument("--start_date", default="2023-08-01")
    parser.add_argument("--train_start", default="2023-08-01")
    parser.add_argument("--val_start", default="2023-09-01")
    parser.add_argument("--test_start", default="2023-09-11")
    parser.add_argument("--end_date", default="2023-09-30")
    parser.add_argument("--split_strategy", choices=["calendar", "relative"], default="calendar")
    parser.add_argument("--relative_train_frac", type=float, default=0.6)
    parser.add_argument("--relative_val_frac", type=float, default=0.2)
    parser.add_argument("--ensure_visible_opposite", action="store_true")
    parser.add_argument("--min_rating", type=float, default=4.0)
    parser.add_argument("--verified_only", action="store_true")
    parser.add_argument("--min_user_count", type=int, default=2)
    parser.add_argument("--min_item_count", type=int, default=1)
    parser.add_argument("--warmup_k", type=int, default=5)
    parser.add_argument("--content_dim", type=int, default=64)
    parser.add_argument("--max_raw_reviews", type=int, default=0)
    parser.add_argument("--max_eval_entities", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--progress_every", type=int, default=100000)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    out_dir = args.root / "data" / args.dataset
    if out_dir.exists() and not args.overwrite:
        raise FileExistsError(f"{out_dir} exists; pass --overwrite to rebuild")
    out_dir.mkdir(parents=True, exist_ok=True)

    reviews = collect_reviews(args)
    meta = collect_metadata(args, set(reviews["raw_item"].unique().tolist()))
    reviews = reviews[reviews["raw_item"].isin(set(meta["raw_item"]))].copy()
    reviews = iterative_filter(reviews, args.min_user_count, args.min_item_count)
    if len(reviews) == 0:
        raise RuntimeError("All reviews were removed by metadata/count filters")

    raw_users = sorted(reviews["raw_user"].unique().tolist())
    raw_items = sorted(reviews["raw_item"].unique().tolist())
    user_map = {raw: idx for idx, raw in enumerate(raw_users)}
    item_map = {raw: idx for idx, raw in enumerate(raw_items)}
    reviews["user"] = reviews["raw_user"].map(user_map).astype(np.int64)
    reviews["item"] = reviews["raw_item"].map(item_map).astype(np.int64)
    meta = meta[meta["raw_item"].isin(item_map)].copy()
    meta["item"] = meta["raw_item"].map(item_map).astype(np.int64)
    meta = meta.sort_values("item").reset_index(drop=True)

    splits = split_temporal(reviews, args)
    split_full = splits["full"]
    if args.max_eval_entities:
        split_full = sample_eval_entities(split_full, args.cold_object, args.max_eval_entities, args.seed)
        splits = split_temporal(split_full.drop(columns=["period"], errors="ignore"), args)

    item_ids = sorted(splits["full"]["item"].unique().tolist())
    keep_items = set(item_ids)
    meta = meta[meta["item"].isin(keep_items)].copy()
    # Remap again after optional sampling so arrays stay compact.
    keep_users = sorted(splits["full"]["user"].unique().tolist())
    user_remap = {old: new for new, old in enumerate(keep_users)}
    item_remap = {old: new for new, old in enumerate(item_ids)}
    for key, value in list(splits.items()):
        if isinstance(value, pd.DataFrame):
            value = value[value["user"].isin(user_remap) & value["item"].isin(item_remap)].copy()
            value["user"] = value["user"].map(user_remap).astype(np.int64)
            value["item"] = value["item"].map(item_remap).astype(np.int64)
            splits[key] = value
    strict_group_df = pd.concat(
        [splits[f"cold_{args.cold_object}_val"], splits[f"cold_{args.cold_object}_test"]],
        ignore_index=True,
    )
    warmup_group_df = pd.concat(
        [splits["warmup_support"], splits["warmup_val"], splits["warmup_test"]],
        ignore_index=True,
    )
    warm_group_df = pd.concat(
        [splits["warm_train_original"], splits["warm_val"], splits["warm_test"]],
        ignore_index=True,
    )
    splits["entity_groups"] = {
        "strict": sorted(strict_group_df[args.cold_object].unique().tolist()) if len(strict_group_df) else [],
        "warmup": sorted(warmup_group_df[args.cold_object].unique().tolist()) if len(warmup_group_df) else [],
        "warm": sorted(warm_group_df[args.cold_object].unique().tolist()) if len(warm_group_df) else [],
    }
    meta["item"] = meta["item"].map(item_remap).astype(np.int64)
    user_num = len(user_remap)
    item_num = len(item_remap)

    for filename, df in [
        (f"{args.dataset}.csv", splits["full"]),
        ("warm_emb.csv", splits["train"]),
        ("warm_train.csv", splits["train"]),
        ("warm_emb_original.csv", splits["warm_train_original"]),
        ("warm_train_original.csv", splits["warm_train_original"]),
        ("warm_val.csv", splits["warm_val"]),
        ("warm_test.csv", splits["warm_test"]),
        ("warmup_support.csv", splits["warmup_support"]),
        ("warmup_val.csv", splits["warmup_val"]),
        ("warmup_test.csv", splits["warmup_test"]),
        (f"cold_{args.cold_object}.csv", splits[f"cold_{args.cold_object}"]),
        (f"cold_{args.cold_object}_val.csv", splits[f"cold_{args.cold_object}_val"]),
        (f"cold_{args.cold_object}_test.csv", splits[f"cold_{args.cold_object}_test"]),
        ("overall_val.csv", splits["overall_val"]),
        ("overall_test.csv", splits["overall_test"]),
    ]:
        write_pairs(df, out_dir / filename)

    raw_data = meta[
        ["item", "raw_item", "title", "categories", "features", "description", "details", "average_rating", "rating_number", "metadata_text"]
    ].sort_values("item")
    raw_data.to_csv(out_dir / "raw-data.csv", index=False)
    mapping = {
        "user_map": {raw: int(user_remap[mapped]) for raw, mapped in user_map.items() if mapped in user_remap},
        "item_map": {raw: int(item_remap[mapped]) for raw, mapped in item_map.items() if mapped in item_remap},
    }
    (out_dir / "id_mapping.json").write_text(json.dumps(mapping, indent=2), encoding="utf-8")
    content_path = out_dir / f"{args.dataset}_{args.cold_object}_content.npy"
    build_content(raw_data, item_num, content_path, args.content_dim)
    write_convert_dict(out_dir, args.dataset, args.cold_object, splits, user_num, item_num, args.warmup_k)

    split_meta = {
        "source": "Amazon Reviews 2023 raw review_categories/meta_categories",
        "category": args.category,
        "dataset": args.dataset,
        "cold_object": args.cold_object,
        "post_cutoff_window": {"start": args.start_date, "end": args.end_date},
        "temporal_split": {
            "strategy": args.split_strategy,
            "train": [args.train_start, args.val_start],
            "validation": [args.val_start, args.test_start],
            "test": [args.test_start, args.end_date],
            "relative_train_frac": args.relative_train_frac,
            "relative_val_frac": args.relative_val_frac,
            "ensure_visible_opposite": bool(args.ensure_visible_opposite),
        },
        "min_rating": args.min_rating,
        "verified_only": bool(args.verified_only),
        "min_user_count": args.min_user_count,
        "min_item_count": args.min_item_count,
        "warmup_k": args.warmup_k,
        "max_eval_entities": args.max_eval_entities,
        "user_num": user_num,
        "item_num": item_num,
        "splits": {
            name: count_stats(splits[name])
            for name in [
                "train",
                "warm_train_original",
                "warm_val",
                "warm_test",
                "warmup_support",
                "warmup_val",
                "warmup_test",
                f"cold_{args.cold_object}_val",
                f"cold_{args.cold_object}_test",
                "overall_val",
                "overall_test",
            ]
        },
        "entity_groups": {key: len(value) for key, value in splits["entity_groups"].items()},
        "content_path": str(content_path),
    }
    (out_dir / "split_meta.json").write_text(json.dumps(split_meta, indent=2), encoding="utf-8")
    print(json.dumps(split_meta, indent=2), flush=True)


if __name__ == "__main__":
    main()
