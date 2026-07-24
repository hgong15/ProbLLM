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
MODELS="${MODELS:-lgn}"

run_variant() {
  local dataset="$1"
  local experiment_name="$2"
  local model="$3"
  local method="$4"
  local seed="$5"
  local extended_csv="$6"

  case "$method" in
    probllm_wo_caga)
      RWFT_WEIGHTED=1 RWFT_BETA=0.8 CAGA_GAMMA=0.0 CAGA_K0="$WARMUP_K" WARMUP_K="$WARMUP_K" \
        bash "$ROOT/scripts/run_revision_finalupdate.sh" "$dataset" item "$model" "$method" "$seed" "$extended_csv" 100
      ;;
    probllm_wo_rwft)
      RWFT_WEIGHTED=1 RWFT_BETA=1.0 CAGA_GAMMA=1.0 CAGA_K0="$WARMUP_K" SIM_PROB_COLUMN="__hard_label_probability__" WARMUP_K="$WARMUP_K" \
        bash "$ROOT/scripts/run_revision_finalupdate.sh" "$dataset" item "$model" "$method" "$seed" "$extended_csv" 100
      ;;
    probllm_wo_rwft_caga)
      RWFT_WEIGHTED=0 RWFT_BETA=1.0 CAGA_GAMMA=0.0 CAGA_K0="$WARMUP_K" WARMUP_K="$WARMUP_K" \
        bash "$ROOT/scripts/run_revision_finalupdate.sh" "$dataset" item "$model" "$method" "$seed" "$extended_csv" 100
      ;;
    *)
      echo "Unknown optimization ablation method: $method" >&2
      exit 1
      ;;
  esac
}

for dataset in $DATASETS; do
  case "$dataset" in
    CiteULike)
      experiment_name="CiteULike_item"
      ;;
    ml-1m)
      experiment_name="ml-1m_item"
      ;;
    *)
      echo "Unsupported dataset in DATASETS: $dataset" >&2
      exit 1
      ;;
  esac

  for model in $MODELS; do
    for method in probllm_wo_caga probllm_wo_rwft probllm_wo_rwft_caga; do
      for seed in "${SEEDS[@]}"; do
        extended_csv="$ROOT/experiments/multiseed/${experiment_name}/seed_${seed}/predicted_cold_item_interaction.csv"
        if [[ ! -s "$extended_csv" ]]; then
          echo "Missing ProbLLM extended file: $extended_csv" >&2
          exit 1
        fi

        if [[ -s "$ROOT/experiments/revision_main/${experiment_name}/${model}/${method}/seed_${seed}/final_metrics.json" ]]; then
          echo "SKIP $experiment_name $model $method seed=$seed"
          continue
        fi

        run_variant "$dataset" "$experiment_name" "$model" "$method" "$seed" "$extended_csv"
      done

      "$CONDA_EXE" run --no-capture-output -n "$PROBLLM_ENV" \
        python -u "$ROOT/scripts/aggregate_multiseed_results.py" \
          --experiment_root "$ROOT/experiments/revision_main/${experiment_name}/${model}/${method}" \
          --method_name "$method $model $dataset" \
          --seeds "${SEEDS[@]}"
    done
  done
done
