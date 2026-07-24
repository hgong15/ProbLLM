import argparse
import csv
import json
import re
import statistics
from pathlib import Path


NONLLM_BEST = {
    "overall": {
        "recall@20": 0.149310,
        "ndcg@20": 0.166096,
        "best_method": "DeepMusic",
    },
    "strict_cold": {
        "recall@20": 0.289214,
        "ndcg@20": 0.183930,
        "best_method": "Heater",
    },
    "warmup": {
        "recall@20": 0.272538,
        "ndcg@20": 0.179220,
        "best_method": "Heater",
    },
    "warm": {
        "recall@20": 0.325343,
        "ndcg@20": 0.251620,
        "best_method": "GNP",
    },
}


VAL_NDCG_RE = re.compile(r"\[BEST-CHECK\].*mode=val\s+overall_ndcg@20=([0-9.]+)")
VAL_OVERALL_RE = re.compile(
    r"Overall Result: .*?'recall': array\(\[([0-9.]+)\]\).*?'ndcg': array\(\[([0-9.]+)\]\)"
)


def parse_val_metrics(log_path):
    text = log_path.read_text(errors="ignore")
    val_ndcg_matches = VAL_NDCG_RE.findall(text)
    val_ndcg = float(val_ndcg_matches[-1]) if val_ndcg_matches else None
    val_recall = None
    lines = text.splitlines()
    for idx, line in enumerate(lines):
        if "[VAL]" not in line:
            continue
        for candidate in lines[idx + 1 : idx + 30]:
            if "Overall Result:" not in candidate:
                continue
            match = VAL_OVERALL_RE.search(candidate)
            if match:
                val_recall = float(match.group(1))
                val_ndcg = float(match.group(2))
            break
    return {"val_recall@20": val_recall, "val_ndcg@20": val_ndcg}


def metric_mean(values):
    return sum(values) / len(values) if values else None


def metric_std(values):
    if len(values) < 2:
        return 0.0
    return statistics.stdev(values)


def collect_rows(root, method_glob, seeds):
    rows = []
    exp_root = root / "experiments" / "revision_main" / "ml-1m_item" / "lgn"
    for method_dir in sorted(exp_root.glob(method_glob)):
        if not method_dir.is_dir():
            continue
        method = method_dir.name
        for seed in seeds:
            seed_dir = method_dir / f"seed_{seed}"
            metrics_path = seed_dir / "final_metrics.json"
            log_path = seed_dir / "finalupdate_from_backbone.log"
            if not metrics_path.exists() or not log_path.exists():
                continue
            try:
                metrics = json.loads(metrics_path.read_text())
                test = metrics["test"]
            except Exception:
                continue
            val = parse_val_metrics(log_path)
            if val["val_ndcg@20"] is None:
                continue
            rows.append(
                {
                    "seed": seed,
                    "method": method,
                    "metrics_path": str(metrics_path),
                    "log_path": str(log_path),
                    **val,
                    "test": test,
                }
            )
    return rows


def select_rows(rows, seeds, selection_metric):
    selected = []
    for seed in seeds:
        candidates = [row for row in rows if row["seed"] == seed and row.get(selection_metric) is not None]
        if not candidates:
            raise RuntimeError(f"No completed candidates for seed {seed}")
        selected.append(max(candidates, key=lambda row: row[selection_metric]))
    return selected


def summarize(selected):
    splits = {}
    for split in ["overall", "strict_cold", "warmup", "warm"]:
        split_summary = {}
        for metric in ["recall@20", "ndcg@20"]:
            values = [float(row["test"][split][metric]) for row in selected]
            mean = metric_mean(values)
            best = NONLLM_BEST[split][metric]
            split_summary[metric] = {
                "mean": mean,
                "std": metric_std(values),
                "values": values,
                "nonllm_best": best,
                "nonllm_best_method": NONLLM_BEST[split]["best_method"],
                "improvement": mean / best - 1.0,
                "meets_5pct": mean >= 1.05 * best,
            }
        splits[split] = split_summary
    main_splits = ["overall", "strict_cold", "warmup"]
    meets_main = all(
        splits[split][metric]["meets_5pct"]
        for split in main_splits
        for metric in ["recall@20", "ndcg@20"]
    )
    return {"splits": splits, "meets_5pct_main_table": meets_main}


def write_outputs(output_dir, root, selected, rows, selection_metric, summary):
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset": "ml-1m",
        "cold_object": "item",
        "model": "LightGCN",
        "selection": {
            "metric": selection_metric,
            "candidate_method_glob": "probllm_tuned_top100_*_5seed",
            "num_completed_seed_method_rows": len(rows),
            "note": "For each seed, the ProbLLM topN/alpha/prior-shape variant is selected by validation overall metrics; reported metrics are held-out test metrics.",
        },
        "seeds": [row["seed"] for row in selected],
        "selected": [
            {
                "seed": row["seed"],
                "method": row["method"],
                "val_recall@20": row["val_recall@20"],
                "val_ndcg@20": row["val_ndcg@20"],
                "metrics_path": row["metrics_path"],
                "log_path": row["log_path"],
                "test": row["test"],
            }
            for row in selected
        ],
        **summary,
    }
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    with (output_dir / "selected_methods.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "seed",
                "method",
                "val_recall@20",
                "val_ndcg@20",
                "overall_recall@20",
                "overall_ndcg@20",
                "strict_cold_recall@20",
                "strict_cold_ndcg@20",
                "warmup_recall@20",
                "warmup_ndcg@20",
            ]
        )
        for row in selected:
            writer.writerow(
                [
                    row["seed"],
                    row["method"],
                    row["val_recall@20"],
                    row["val_ndcg@20"],
                    row["test"]["overall"]["recall@20"],
                    row["test"]["overall"]["ndcg@20"],
                    row["test"]["strict_cold"]["recall@20"],
                    row["test"]["strict_cold"]["ndcg@20"],
                    row["test"]["warmup"]["recall@20"],
                    row["test"]["warmup"]["ndcg@20"],
                ]
            )

    lines = []
    lines.append("# MovieLens Item-Cold ProbLLM Val-Selected 5Seed")
    lines.append("")
    lines.append(f"- root: `{root}`")
    lines.append(f"- selection metric: `{selection_metric}`")
    lines.append(f"- completed seed-method candidates: `{len(rows)}`")
    lines.append(f"- meets main-table +5% target: `{summary['meets_5pct_main_table']}`")
    lines.append("")
    lines.append("## Selected Configs")
    lines.append("")
    lines.append("| Seed | Selected Method | Val R/N | Test Overall R/N |")
    lines.append("|---:|---|---:|---:|")
    for row in selected:
        lines.append(
            "| {seed} | `{method}` | {vr:.6f} / {vn:.6f} | {tr:.6f} / {tn:.6f} |".format(
                seed=row["seed"],
                method=row["method"],
                vr=row["val_recall@20"],
                vn=row["val_ndcg@20"],
                tr=row["test"]["overall"]["recall@20"],
                tn=row["test"]["overall"]["ndcg@20"],
            )
        )
    lines.append("")
    lines.append("## Test Mean")
    lines.append("")
    lines.append("| Split | Mean R/N | SD R/N | Best NonLLM R/N | Improvement R/N | Meets +5% |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for split in ["overall", "strict_cold", "warmup", "warm"]:
        r = summary["splits"][split]["recall@20"]
        n = summary["splits"][split]["ndcg@20"]
        lines.append(
            "| {split} | {rm:.6f} / {nm:.6f} | {rs:.6f} / {ns:.6f} | {rb:.6f} / {nb:.6f} ({bm}) | {ri:.2f}% / {ni:.2f}% | {meet} |".format(
                split=split,
                rm=r["mean"],
                nm=n["mean"],
                rs=r["std"],
                ns=n["std"],
                rb=r["nonllm_best"],
                nb=n["nonllm_best"],
                bm=r["nonllm_best_method"],
                ri=100.0 * r["improvement"],
                ni=100.0 * n["improvement"],
                meet=r["meets_5pct"] and n["meets_5pct"],
            )
        )
    lines.append("")
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--seeds", nargs="+", type=int, default=[2020, 2021, 2022, 2023, 42])
    parser.add_argument("--method_glob", default="probllm_tuned_top100_*_5seed")
    parser.add_argument("--selection_metric", choices=["val_ndcg@20", "val_recall@20"], default="val_ndcg@20")
    parser.add_argument("--output_dir", default="results/ml1m_item_probllm_val_selected_5seed")
    args = parser.parse_args()

    root = Path(args.root)
    rows = collect_rows(root, args.method_glob, args.seeds)
    selected = select_rows(rows, args.seeds, args.selection_metric)
    summary = summarize(selected)
    write_outputs(Path(args.output_dir), root, selected, rows, args.selection_metric, summary)
    print((Path(args.output_dir) / "summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
