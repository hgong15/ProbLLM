#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-.}"
PYTHON_BIN="${PYTHON_BIN:-python}"
LLAMA_FACTORY_DIR="${LLAMA_FACTORY_DIR:-../LLaMA-Factory}"
BASE_7B="${BASE_7B:-../models/Llama-2-7b-hf}"
LORA_7B_TEMPLATE="${LORA_7B_TEMPLATE:-$LLAMA_FACTORY_DIR/examples/train_lora/llama2_lora_sft_probllm.yaml}"
SEEDS=(42 2020 2021 2022 2023)
K_VALUES="${K_VALUES:-3 10}"
DATASETS="${DATASETS:-CiteULike ml-1m}"
FORCE_K0_RERUN="${FORCE_K0_RERUN:-1}"

run_seed_pipeline() {
  local dataset="$1"
  local dataset_key="$2"
  local seed="$3"
  local k0="$4"
  local experiment_name="${dataset_key}_llama2-7b_k0_${k0}"
  local seed_dir="$ROOT/experiments/multiseed/${experiment_name}/seed_${seed}"
  local pred="$seed_dir/predicted_cold_item_interaction.csv"
  local log_file="$seed_dir/train_logs.txt"

  if [[ "$FORCE_K0_RERUN" != "1" && -s "$pred" ]]; then
    echo "SKIP prediction $experiment_name seed=$seed"
    return
  fi

  mkdir -p "$seed_dir"
  case "$dataset" in
    CiteULike)
      BASE_MODEL="$BASE_7B" \
      MODEL_TEMPLATE=llama2 \
      LORA_TEMPLATE="$LORA_7B_TEMPLATE" \
      WARMUP_K="$k0" \
      EXPERIMENT_NAME="$experiment_name" \
      MODEL_TAG="llama2-7b-citeulike-k0${k0}" \
      HF_INFER_BATCH_SIZE=4 \
      LLM_EMBED_BATCH_SIZE=4 \
      START_STEP=1 END_STEP=9 \
        bash "$ROOT/scripts/run_seed_citeulike_item.sh" "$seed"
      "$PYTHON_BIN" -u "$ROOT/scripts/summarize_seed_outputs.py" \
        --seed "$seed" \
        --data_dir "$ROOT/data/CiteULike" \
        --seed_dir "$seed_dir" \
        --log_file "$log_file"
      ;;
    ml-1m)
      BASE_MODEL="$BASE_7B" \
      MODEL_TEMPLATE=llama2 \
      LORA_TEMPLATE="$LORA_7B_TEMPLATE" \
      WARMUP_K="$k0" \
      EXPERIMENT_NAME="$experiment_name" \
      MODEL_TAG="llama2-7b-ml1m-k0${k0}" \
      HF_INFER_BATCH_SIZE=4 \
      LLM_EMBED_BATCH_SIZE=4 \
      START_STEP=1 END_STEP=9 \
        bash "$ROOT/scripts/run_seed_ml1m_item.sh" "$seed"
      "$PYTHON_BIN" -u "$ROOT/scripts/summarize_seed_outputs.py" \
        --seed "$seed" \
        --data_dir "$ROOT/data/ml-1m" \
        --seed_dir "$seed_dir" \
        --log_file "$log_file"
      ;;
    *)
      echo "Unsupported dataset: $dataset" >&2
      exit 1
      ;;
  esac

  if [[ ! -s "$pred" ]]; then
    echo "Prediction was not generated: $pred" >&2
    exit 1
  fi
}

run_downstream() {
  local dataset="$1"
  local dataset_key="$2"
  local seed="$3"
  local k0="$4"
  local experiment_name="${dataset_key}_llama2-7b_k0_${k0}"
  local method="probllm_k0_${k0}"
  local pred="$ROOT/experiments/multiseed/${experiment_name}/seed_${seed}/predicted_cold_item_interaction.csv"
  local final_metrics="$ROOT/experiments/revision_main/${dataset_key}/lgn/${method}/seed_${seed}/final_metrics.json"

  if [[ "$FORCE_K0_RERUN" != "1" && -s "$final_metrics" ]]; then
    echo "SKIP downstream $dataset_key $method seed=$seed"
    return
  fi
  if [[ ! -s "$pred" ]]; then
    echo "Missing prediction for downstream: $pred" >&2
    exit 1
  fi

  RWFT_WEIGHTED=1 \
  RWFT_BETA=0.8 \
  CAGA_GAMMA=1.0 \
  CAGA_K0="$k0" \
  WARMUP_K="$k0" \
    bash "$ROOT/scripts/run_revision_finalupdate.sh" \
      "$dataset" item lgn "$method" "$seed" "$pred" 100
}

aggregate_method() {
  local dataset_key="$1"
  local k0="$2"
  local method="probllm_k0_${k0}"
  "$PYTHON_BIN" -u "$ROOT/scripts/aggregate_multiseed_results.py" \
    --experiment_root "$ROOT/experiments/revision_main/${dataset_key}/lgn/${method}" \
    --method_name "ProbLLM k0=${k0} LightGCN ${dataset_key}" \
    --seeds "${SEEDS[@]}"
}

for dataset in $DATASETS; do
  case "$dataset" in
    CiteULike) dataset_key="CiteULike_item" ;;
    ml-1m) dataset_key="ml-1m_item" ;;
    *) echo "Unsupported dataset in DATASETS: $dataset" >&2; exit 1 ;;
  esac

  for k0 in $K_VALUES; do
    for seed in "${SEEDS[@]}"; do
      run_seed_pipeline "$dataset" "$dataset_key" "$seed" "$k0"
      run_downstream "$dataset" "$dataset_key" "$seed" "$k0"
    done
    aggregate_method "$dataset_key" "$k0"
  done
done
