import argparse
import json
import pickle
import re
from pathlib import Path

import pandas as pd


def parse_probability(value):
    if value is None:
        return None
    if isinstance(value, (int, float)):
        prob = float(value)
    else:
        text = str(value).strip()
        try:
            prob = float(text)
        except ValueError:
            match = re.search(r"(?<![\d.])(?:0\.\d+|1(?:\.0+)?|0)(?!\d)", text)
            if not match:
                return None
            prob = float(match.group(0))
    if prob < 0 or prob > 1:
        return None
    return prob


def read_predictions(path):
    predictions = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                predictions.append(None)
                continue
            predictions.append(parse_probability(row.get("predict")))
    return predictions


def read_raw_titles(data_dir, dataset):
    raw_path = Path(data_dir) / "raw-data.csv"
    encoding = "latin1" if dataset == "CiteULike" else "utf-8"
    raw = pd.read_csv(raw_path, encoding=encoding)
    if "title" not in raw.columns:
        title_col = next((col for col in raw.columns if "title" in col.lower()), None)
        if title_col is None:
            raw["title"] = raw.index.map(lambda idx: f"item {idx}")
        else:
            raw["title"] = raw[title_col]
    return raw["title"].fillna("").astype(str).tolist()


def read_user_history(data_dir):
    pref_path = Path(data_dir) / "train_user_preference_list.pkl"
    with pref_path.open("rb") as f:
        prefs = pickle.load(f)
    return prefs


def get_history(prefs, user_id):
    try:
        value = prefs[user_id]
    except Exception:
        value = ""
    if isinstance(value, (list, tuple)):
        value = ", ".join(map(str, value))
    return str(value)


def truncate(text, limit):
    text = " ".join(str(text).split())
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)] + "..."


def read_truth_pairs(data_dir, cold_object):
    truth = set()
    split_names = [
        f"cold_{cold_object}_test.csv",
        "warmup_test.csv",
        "overall_test.csv",
    ]
    for name in split_names:
        path = Path(data_dir) / name
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if {"user", "item"}.issubset(df.columns):
            truth.update((int(row.user), int(row.item)) for row in df.itertuples(index=False))
    return truth


def select_cases(scored, n):
    cases = []
    selectors = [
        ("TP_high", scored[(scored.is_heldout) & (scored.probability.notna())].sort_values("probability", ascending=False)),
        ("FP_high", scored[(~scored.is_heldout) & (scored.probability.notna())].sort_values("probability", ascending=False)),
        ("FN_low", scored[(scored.is_heldout) & (scored.probability.notna())].sort_values("probability", ascending=True)),
    ]
    for case_type, frame in selectors:
        take = frame.head(n).copy()
        take["case_type"] = case_type
        cases.append(take)
    return pd.concat(cases, ignore_index=True) if cases else pd.DataFrame()


def write_markdown(path, selected):
    lines = ["# Qualitative LLM Simulation Cases", ""]
    for case_type, group in selected.groupby("case_type", sort=False):
        lines.extend([f"## {case_type}", ""])
        for row in group.itertuples(index=False):
            lines.append(
                f"- user={row.user}, item={row.item}, group={row.entity_type}, "
                f"prob={row.probability:.6f}, heldout={row.is_heldout}"
            )
            lines.append(f"  - target: {row.item_title_short}")
            lines.append(f"  - history: {row.user_history_short}")
        lines.append("")
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Build qualitative TP/FP/FN cases for LLM-simulated interactions.")
    parser.add_argument("--dataset", required=True, choices=["CiteULike", "ml-1m"])
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--top20_csv", required=True)
    parser.add_argument("--prediction_jsonl", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--cold_object", default="item", choices=["item", "user"])
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--history_chars", type=int, default=360)
    parser.add_argument("--title_chars", type=int, default=180)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    top20 = pd.read_csv(args.top20_csv)
    predictions = read_predictions(args.prediction_jsonl)
    usable_len = min(len(top20), len(predictions))
    if usable_len == 0:
        raise SystemExit("No aligned candidates/predictions found.")

    scored = top20.iloc[:usable_len].copy().reset_index(drop=True)
    scored["probability"] = predictions[:usable_len]
    scored["user"] = scored["user"].astype(int)
    scored["item"] = scored["item"].astype(int)
    if "entity_type" not in scored.columns:
        scored["entity_type"] = ""

    truth_pairs = read_truth_pairs(data_dir, args.cold_object)
    titles = read_raw_titles(data_dir, args.dataset)
    prefs = read_user_history(data_dir)

    scored["is_heldout"] = [(int(u), int(i)) in truth_pairs for u, i in zip(scored.user, scored.item)]
    scored["item_title"] = [titles[i] if 0 <= i < len(titles) else f"item {i}" for i in scored.item]
    scored["user_history"] = [get_history(prefs, int(u)) for u in scored.user]
    scored["item_title_short"] = scored["item_title"].map(lambda text: truncate(text, args.title_chars))
    scored["user_history_short"] = scored["user_history"].map(lambda text: truncate(text, args.history_chars))

    scored_path = output_dir / f"{args.dataset}_scored_cases.csv"
    selected_path = output_dir / f"{args.dataset}_selected_cases.csv"
    md_path = output_dir / f"{args.dataset}_selected_cases.md"
    summary_path = output_dir / f"{args.dataset}_case_summary.json"

    selected = select_cases(scored, args.n)
    scored.to_csv(scored_path, index=False)
    selected.to_csv(selected_path, index=False)
    write_markdown(md_path, selected)

    summary = {
        "dataset": args.dataset,
        "top20_rows": int(len(top20)),
        "prediction_rows": int(len(predictions)),
        "aligned_rows": int(usable_len),
        "heldout_positive_candidates": int(scored["is_heldout"].sum()),
        "selected_cases": int(len(selected)),
        "scored_cases_csv": str(scored_path),
        "selected_cases_csv": str(selected_path),
        "selected_cases_md": str(md_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
