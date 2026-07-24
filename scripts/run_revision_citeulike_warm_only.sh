#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-.}"
CONDA_EXE="${CONDA_EXE:-conda}"
PROBLLM_ENV="${PROBLLM_ENV:-probllm}"
SEEDS=("${@:-42 2020 2021 2022 2023}")
EMPTY_CSV="$ROOT/experiments/revision_protocol/empty_interactions.csv"

if [[ "$#" -eq 0 ]]; then
  SEEDS=(42 2020 2021 2022 2023)
fi

for model in mf lgn; do
  for seed in "${SEEDS[@]}"; do
    bash "$ROOT/scripts/run_revision_finalupdate.sh" \
      CiteULike \
      item \
      "$model" \
      warm_only \
      "$seed" \
      "$EMPTY_CSV" \
      100
  done

  "$CONDA_EXE" run --no-capture-output -n "$PROBLLM_ENV" \
    python -u "$ROOT/scripts/aggregate_multiseed_results.py" \
      --experiment_root "$ROOT/experiments/revision_main/CiteULike_item/$model/warm_only" \
      --method_name "Warm-only $model" \
      --seeds "${SEEDS[@]}"
done
