#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-.}"
CONDA_EXE="${CONDA_EXE:-conda}"
PROBLLM_ENV="${PROBLLM_ENV:-probllm}"

if [[ "$#" -eq 0 ]]; then
  SEEDS=(42 2020 2021 2022 2023)
else
  SEEDS=("$@")
fi

for model in mf lgn; do
  for seed in "${SEEDS[@]}"; do
    EXTENDED_CSV="$ROOT/experiments/multiseed/CiteULike_item/seed_${seed}/predicted_cold_item_interaction.csv"
    if [[ ! -s "$EXTENDED_CSV" ]]; then
      echo "Missing ProbLLM extended file: $EXTENDED_CSV" >&2
      exit 1
    fi

    bash "$ROOT/scripts/run_revision_finalupdate.sh" \
      CiteULike \
      item \
      "$model" \
      probllm \
      "$seed" \
      "$EXTENDED_CSV" \
      100
  done

  "$CONDA_EXE" run --no-capture-output -n "$PROBLLM_ENV" \
    python -u "$ROOT/scripts/aggregate_multiseed_results.py" \
      --experiment_root "$ROOT/experiments/revision_main/CiteULike_item/$model/probllm" \
      --method_name "ProbLLM $model" \
      --seeds "${SEEDS[@]}"
done
