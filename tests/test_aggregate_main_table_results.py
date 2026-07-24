import csv
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from scipy.stats import ttest_rel


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "aggregate_main_table_results.py"
SPEC = importlib.util.spec_from_file_location("aggregate_main_table_results", SCRIPT)
AGGREGATE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(AGGREGATE)


class AggregateMainTableResultsTest(unittest.TestCase):
    def test_paired_test_is_two_sided(self):
        target = [2.0, 4.0, 6.0, 8.0, 10.0]
        comparator = [1.0, 2.0, 3.0, 4.0, 5.0]
        expected = float(ttest_rel(target, comparator).pvalue)
        one_sided = float(ttest_rel(target, comparator, alternative="greater").pvalue)

        self.assertAlmostEqual(AGGREGATE.paired_test(target, comparator), expected)
        self.assertAlmostEqual(expected, 2.0 * one_sided)

    def test_loads_predeclared_cellwise_comparator_map(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "comparators.json"
            path.write_text(
                json.dumps({"overall": {"recall@20": "Baseline"}}),
                encoding="utf-8",
            )
            self.assertEqual(
                AGGREGATE.load_comparator_map(path),
                {("overall", "recall@20"): "Baseline"},
            )

    def test_cli_requires_explicit_comparator_and_writes_two_sided_pvalue(self):
        seeds = [42, 2020, 2021, 2022, 2023]
        target_values = [2.0, 4.0, 6.0, 8.0, 10.0]
        baseline_values = [1.0, 2.0, 3.0, 4.0, 5.0]

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target_root = root / "target"
            baseline_root = root / "baseline"
            output_root = root / "output"
            for seed, target, baseline in zip(seeds, target_values, baseline_values):
                for method_root, value in (
                    (target_root, target),
                    (baseline_root, baseline),
                ):
                    seed_root = method_root / f"seed_{seed}"
                    seed_root.mkdir(parents=True)
                    (seed_root / "final_metrics.json").write_text(
                        json.dumps({"test": {"overall": {"recall@20": value}}}),
                        encoding="utf-8",
                    )

            command = [
                sys.executable,
                str(SCRIPT),
                "--method",
                f"ProbLLM={target_root}",
                "--method",
                f"Baseline={baseline_root}",
                "--target",
                "ProbLLM",
                "--comparator",
                "Baseline",
                "--splits",
                "overall",
                "--metrics",
                "recall@20",
                "--output-dir",
                str(output_root),
            ]
            subprocess.run(command, check=True, capture_output=True, text=True)

            with (output_root / "paired_tests.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                row = next(csv.DictReader(handle))
            expected = float(ttest_rel(target_values, baseline_values).pvalue)
            self.assertAlmostEqual(float(row["two_sided_pvalue_unadjusted"]), expected)
            self.assertEqual(row["comparator"], "Baseline")

            command_without_comparator = command.copy()
            comparator_index = command_without_comparator.index("--comparator")
            del command_without_comparator[comparator_index : comparator_index + 2]
            failed = subprocess.run(
                command_without_comparator, capture_output=True, text=True
            )
            self.assertNotEqual(failed.returncode, 0)
            self.assertIn(
                "one of the arguments --comparator --comparator-map is required",
                failed.stderr,
            )


if __name__ == "__main__":
    unittest.main()
