#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-.}"
CONDA_EXE="${CONDA_EXE:-conda}"
PROBLLM_ENV="${PROBLLM_ENV:-probllm}"
WARMUP_K="${WARMUP_K:-5}"

if [[ "$#" -eq 0 ]]; then
  SEEDS=(42 2020 2021 2022 2023)
else
  SEEDS=("$@")
fi

DATASETS="${DATASETS:-CiteULike ml-1m}"
BETA_VALUES="${BETA_VALUES:-0.2 0.5 1.2 1.5}"
GAMMA_VALUES="${GAMMA_VALUES:-0.5 2.0}"

tag_float() {
  local value="$1"
  echo "$value" | sed 's/\./p/g'
}

run_one() {
  local dataset="$1"
  local dataset_key="$2"
  local method="$3"
  local seed="$4"
  local beta="$5"
  local gamma="$6"
  local extended_csv="$ROOT/experiments/multiseed/${dataset_key}/seed_${seed}/predicted_cold_item_interaction.csv"

  if [[ ! -s "$extended_csv" ]]; then
    echo "Missing ProbLLM extended file: $extended_csv" >&2
    exit 1
  fi

  if [[ -s "$ROOT/experiments/revision_main/${dataset_key}/lgn/${method}/seed_${seed}/final_metrics.json" ]]; then
    echo "SKIP $dataset_key lgn $method seed=$seed"
    return
  fi

  RWFT_WEIGHTED=1 \
  RWFT_BETA="$beta" \
  CAGA_GAMMA="$gamma" \
  CAGA_K0="$WARMUP_K" \
  WARMUP_K="$WARMUP_K" \
    bash "$ROOT/scripts/run_revision_finalupdate.sh" \
      "$dataset" \
      item \
      lgn \
      "$method" \
      "$seed" \
      "$extended_csv" \
      100
}

for dataset in $DATASETS; do
  case "$dataset" in
    CiteULike)
      dataset_key="CiteULike_item"
      ;;
    ml-1m)
      dataset_key="ml-1m_item"
      ;;
    *)
      echo "Unsupported dataset in DATASETS: $dataset" >&2
      exit 1
      ;;
  esac

  for beta in $BETA_VALUES; do
    beta_tag="$(tag_float "$beta")"
    method="sensitivity_beta_${beta_tag}"
    for seed in "${SEEDS[@]}"; do
      run_one "$dataset" "$dataset_key" "$method" "$seed" "$beta" "1.0"
    done
    "$CONDA_EXE" run --no-capture-output -n "$PROBLLM_ENV" \
      python -u "$ROOT/scripts/aggregate_multiseed_results.py" \
        --experiment_root "$ROOT/experiments/revision_main/${dataset_key}/lgn/${method}" \
        --method_name "ProbLLM beta=${beta} gamma=1.0 LightGCN ${dataset}" \
        --seeds "${SEEDS[@]}"
  done

  for gamma in $GAMMA_VALUES; do
    gamma_tag="$(tag_float "$gamma")"
    method="sensitivity_gamma_${gamma_tag}"
    for seed in "${SEEDS[@]}"; do
      run_one "$dataset" "$dataset_key" "$method" "$seed" "0.8" "$gamma"
    done
    "$CONDA_EXE" run --no-capture-output -n "$PROBLLM_ENV" \
      python -u "$ROOT/scripts/aggregate_multiseed_results.py" \
        --experiment_root "$ROOT/experiments/revision_main/${dataset_key}/lgn/${method}" \
        --method_name "ProbLLM beta=0.8 gamma=${gamma} LightGCN ${dataset}" \
        --seeds "${SEEDS[@]}"
  done
done
