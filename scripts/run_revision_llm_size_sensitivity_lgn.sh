#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-.}"
PYTHON_BIN="${PYTHON_BIN:-python}"
LLAMA_FACTORY_DIR="${LLAMA_FACTORY_DIR:-../LLaMA-Factory}"
BASE_7B="${BASE_7B:-../models/Llama-2-7b-hf}"
LORA_7B_TEMPLATE="${LORA_7B_TEMPLATE:-$LLAMA_FACTORY_DIR/examples/train_lora/llama2_lora_sft_probllm.yaml}"
SEEDS=(42 2020 2021 2022 2023)
DATASETS="${DATASETS:-CiteULike ml-1m}"

run_seed_7b() {
  local dataset="$1"
  local dataset_key="$2"
  local seed="$3"
  local experiment_name="${dataset_key}_llama2-7b"
  local seed_dir="$ROOT/experiments/multiseed/${experiment_name}/seed_${seed}"
  local pred="$seed_dir/predicted_cold_item_interaction.csv"
  local log_file="$seed_dir/train_logs.txt"

  if [[ -s "$pred" ]]; then
    echo "SKIP 7B prediction $experiment_name seed=$seed"
    return
  fi

  mkdir -p "$seed_dir"
  case "$dataset" in
    CiteULike)
      BASE_MODEL="$BASE_7B" \
      MODEL_TEMPLATE=llama2 \
      LORA_TEMPLATE="$LORA_7B_TEMPLATE" \
      EXPERIMENT_NAME="$experiment_name" \
      MODEL_TAG="llama2-7b-citeulike-size" \
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
      EXPERIMENT_NAME="$experiment_name" \
      MODEL_TAG="llama2-7b-ml1m-size" \
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
    echo "7B prediction was not generated: $pred" >&2
    exit 1
  fi
}

run_downstream_7b() {
  local dataset="$1"
  local dataset_key="$2"
  local seed="$3"
  local experiment_name="${dataset_key}_llama2-7b"
  local method="probllm_llama2_7b"
  local pred="$ROOT/experiments/multiseed/${experiment_name}/seed_${seed}/predicted_cold_item_interaction.csv"
  local final_metrics="$ROOT/experiments/revision_main/${dataset_key}/lgn/${method}/seed_${seed}/final_metrics.json"

  if [[ -s "$final_metrics" ]]; then
    echo "SKIP downstream $dataset_key $method seed=$seed"
    return
  fi
  if [[ ! -s "$pred" ]]; then
    echo "Missing 7B prediction for downstream: $pred" >&2
    exit 1
  fi

  RWFT_WEIGHTED=1 \
  RWFT_BETA=0.8 \
  CAGA_GAMMA=1.0 \
  CAGA_K0=5 \
  WARMUP_K=5 \
    bash "$ROOT/scripts/run_revision_finalupdate.sh" \
      "$dataset" item lgn "$method" "$seed" "$pred" 100
}

aggregate_7b() {
  local dataset_key="$1"
  "$PYTHON_BIN" -u "$ROOT/scripts/aggregate_multiseed_results.py" \
    --experiment_root "$ROOT/experiments/revision_main/${dataset_key}/lgn/probllm_llama2_7b" \
    --method_name "ProbLLM LLaMA2-7B LightGCN ${dataset_key}" \
    --seeds "${SEEDS[@]}"
}

for dataset in $DATASETS; do
  case "$dataset" in
    CiteULike) dataset_key="CiteULike_item" ;;
    ml-1m) dataset_key="ml-1m_item" ;;
    *) echo "Unsupported dataset in DATASETS: $dataset" >&2; exit 1 ;;
  esac

  for seed in "${SEEDS[@]}"; do
    run_seed_7b "$dataset" "$dataset_key" "$seed"
    run_downstream_7b "$dataset" "$dataset_key" "$seed"
  done
  aggregate_7b "$dataset_key"
done
