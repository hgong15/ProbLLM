#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-.}"
CONDA_EXE="${CONDA_EXE:-conda}"
PROBLLM_ENV="${PROBLLM_ENV:-probllm}"
NVIDIA_DRIVER_LIB_DIR="${NVIDIA_DRIVER_LIB_DIR:-/usr/local/nvidia/lib64}"
EMPTY_CSV="$ROOT/experiments/revision_protocol/empty_interactions.csv"

if [[ "$#" -eq 0 ]]; then
  SEEDS=(42 2020 2021 2022 2023)
else
  SEEDS=("$@")
fi

if [[ -d "$NVIDIA_DRIVER_LIB_DIR" ]]; then
  export LD_LIBRARY_PATH="$NVIDIA_DRIVER_LIB_DIR:${LD_LIBRARY_PATH:-}"
fi

"$CONDA_EXE" run --no-capture-output -n "$PROBLLM_ENV" \
  python -u "$ROOT/scripts/prepare_ml1m_dataset.py" \
    --data_dir "$ROOT/data/ml-1m" \
    --min_rating 4

for model in mf lgn; do
  for seed in "${SEEDS[@]}"; do
    bash "$ROOT/scripts/run_revision_finalupdate.sh" \
      ml-1m \
      item \
      "$model" \
      warm_only \
      "$seed" \
      "$EMPTY_CSV" \
      100
  done

  "$CONDA_EXE" run --no-capture-output -n "$PROBLLM_ENV" \
    python -u "$ROOT/scripts/aggregate_multiseed_results.py" \
      --experiment_root "$ROOT/experiments/revision_main/ml-1m_item/$model/warm_only" \
      --method_name "Warm-only $model ml-1m" \
      --seeds "${SEEDS[@]}"
done
