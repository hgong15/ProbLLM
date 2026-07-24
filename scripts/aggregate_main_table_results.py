#!/usr/bin/env python3
"""Aggregate matched main-table runs and compute paired significance tests.

Each --method takes NAME=EXPERIMENT_ROOT, where EXPERIMENT_ROOT contains
seed_<seed>/final_metrics.json files produced by summarize_seed_outputs.py.
The comparison method must be declared before aggregation, either once with
--comparator or cell by cell with --comparator-map.  The paper markers use
unadjusted, cellwise, two-sided paired t-tests on matched seeds.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from scipy.stats import ttest_rel


DEFAULT_SEEDS = (42, 2020, 2021, 2022, 2023)
DEFAULT_SPLITS = ("overall", "strict_cold", "warmup")
DEFAULT_METRICS = ("recall@20", "ndcg@20")


def parse_method(value: str) -> Tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--method must be NAME=EXPERIMENT_ROOT")
    name, root = value.split("=", 1)
    if not name or not root:
        raise argparse.ArgumentTypeError("--method must be NAME=EXPERIMENT_ROOT")
    return name, Path(root).expanduser().resolve()


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values)


def sample_std(values: Iterable[float]) -> float:
    values = list(values)
    if len(values) < 2:
        return 0.0
    avg = mean(values)
    return math.sqrt(sum((value - avg) ** 2 for value in values) / (len(values) - 1))


def load_metrics(root: Path, seeds: Iterable[int]) -> Dict[int, dict]:
    output = {}
    for seed in seeds:
        path = root / f"seed_{seed}" / "final_metrics.json"
        if not path.is_file():
            continue
        with path.open(encoding="utf-8") as handle:
            output[seed] = json.load(handle)
    return output


def load_comparator_map(path: Path) -> Dict[Tuple[str, str], str]:
    """Load {split: {metric: method_name}} from a predeclared JSON file."""
    with path.expanduser().resolve().open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("Comparator map must be a JSON object.")

    output: Dict[Tuple[str, str], str] = {}
    for split, metrics in payload.items():
        if not isinstance(split, str) or not isinstance(metrics, dict):
            raise ValueError("Comparator map must have the form {split: {metric: method}}.")
        for metric, method in metrics.items():
            if not isinstance(metric, str) or not isinstance(method, str) or not method:
                raise ValueError("Comparator-map metric names and method names must be strings.")
            output[(split, metric)] = method
    return output


def metric_value(payload: dict, split: str, metric: str) -> float:
    try:
        return float(payload["test"][split][metric])
    except KeyError as exc:
        raise KeyError(f"Missing test/{split}/{metric}") from exc


def write_csv(path: Path, rows: List[dict], fields: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def paired_test(target: List[float], comparator: List[float]) -> float:
    if len(target) < 2:
        return float("nan")
    if all(abs(left - right) < 1e-15 for left, right in zip(target, comparator)):
        return 1.0
    return float(ttest_rel(target, comparator).pvalue)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", action="append", type=parse_method, required=True)
    parser.add_argument("--target", required=True, help="Name of the ProbLLM row to test.")
    comparison = parser.add_mutually_exclusive_group(required=True)
    comparison.add_argument(
        "--comparator",
        help="Predeclared comparison method used for every requested split/metric cell.",
    )
    comparison.add_argument(
        "--comparator-map",
        type=Path,
        help="JSON file of predeclared cellwise methods: {split: {metric: method}}.",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--splits", nargs="+", default=DEFAULT_SPLITS)
    parser.add_argument("--metrics", nargs="+", default=DEFAULT_METRICS)
    args = parser.parse_args()

    methods = dict(args.method)
    if len(methods) != len(args.method):
        parser.error("Method names must be unique.")
    if args.target not in methods:
        parser.error("--target must match one --method name.")
    if args.comparator and args.comparator not in methods:
        parser.error("--comparator must match one --method name.")
    if args.comparator == args.target:
        parser.error("--comparator must differ from --target.")

    try:
        comparator_map = load_comparator_map(args.comparator_map) if args.comparator_map else {}
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(f"Cannot load --comparator-map: {exc}")

    loaded = {name: load_metrics(root, args.seeds) for name, root in methods.items()}
    summary_rows: List[dict] = []
    test_rows: List[dict] = []
    latex_rows: List[str] = []

    for split in args.splits:
        for metric in args.metrics:
            values_by_method = {}
            for name, payloads in loaded.items():
                values_by_method[name] = {
                    seed: metric_value(payload, split, metric)
                    for seed, payload in payloads.items()
                }
                values = list(values_by_method[name].values())
                summary_rows.append(
                    {
                        "method": name,
                        "split": split,
                        "metric": metric,
                        "n": len(values),
                        "mean": f"{mean(values):.8f}" if values else "",
                        "sample_sd": f"{sample_std(values):.8f}" if values else "",
                        "seeds": " ".join(map(str, sorted(values_by_method[name]))),
                    }
                )
                if values:
                    latex_rows.append(
                        f"{name} ({split}, {metric}) & {mean(values):.4f} $\\pm$ {sample_std(values):.4f} \\\\"
                    )

            target_values = values_by_method[args.target]
            comparator_name = args.comparator or comparator_map.get((split, metric))
            if not comparator_name:
                parser.error(
                    f"--comparator-map has no declaration for split={split!r}, metric={metric!r}."
                )
            if comparator_name == args.target:
                parser.error(
                    f"Comparator for split={split!r}, metric={metric!r} must differ from --target."
                )
            if comparator_name not in methods:
                parser.error(
                    f"Comparator {comparator_name!r} for split={split!r}, metric={metric!r} "
                    "does not match a --method name."
                )

            paired_seeds = sorted(set(target_values) & set(values_by_method[comparator_name]))
            missing_seeds = sorted(set(args.seeds) - set(paired_seeds))
            if missing_seeds:
                parser.error(
                    f"Incomplete matched pair for {split}/{metric}: target={args.target!r}, "
                    f"comparator={comparator_name!r}, missing seeds={missing_seeds}."
                )
            target = [target_values[seed] for seed in paired_seeds]
            comparator = [values_by_method[comparator_name][seed] for seed in paired_seeds]
            pvalue = paired_test(target, comparator)
            test_rows.append(
                {
                    "target": args.target,
                    "comparator": comparator_name,
                    "split": split,
                    "metric": metric,
                    "n": len(paired_seeds),
                    "mean_difference": f"{mean(left - right for left, right in zip(target, comparator)):.8f}",
                    "two_sided_pvalue_unadjusted": (
                        "" if math.isnan(pvalue) else f"{pvalue:.8g}"
                    ),
                    "seeds": " ".join(map(str, paired_seeds)),
                    "paper_marker_basis": "unadjusted cellwise two-sided p-value",
                }
            )

    output_dir = args.output_dir.expanduser().resolve()
    write_csv(
        output_dir / "summary_mean_sd.csv",
        summary_rows,
        ["method", "split", "metric", "n", "mean", "sample_sd", "seeds"],
    )
    write_csv(
        output_dir / "paired_tests.csv",
        test_rows,
        [
            "target", "comparator", "split", "metric", "n", "mean_difference",
            "two_sided_pvalue_unadjusted", "seeds", "paper_marker_basis",
        ],
    )
    (output_dir / "latex_mean_sd_rows.tex").write_text("\n".join(latex_rows) + "\n", encoding="utf-8")
    (output_dir / "aggregation_manifest.json").write_text(
        json.dumps(
            {
                "methods": {name: str(path) for name, path in methods.items()},
                "target": args.target,
                "seeds": args.seeds,
                "splits": args.splits,
                "metrics": args.metrics,
                "comparison": {
                    "mode": "fixed method" if args.comparator else "predeclared cellwise map",
                    "value": args.comparator or str(args.comparator_map.expanduser().resolve()),
                },
                "test": "two-sided paired t-test on matched seeds",
                "paper_marker_pvalue": "unadjusted cellwise p-value",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote main-table summaries and paired tests to {output_dir}")


if __name__ == "__main__":
    main()
