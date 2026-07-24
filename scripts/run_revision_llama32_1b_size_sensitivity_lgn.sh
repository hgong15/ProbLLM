#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-.}"
PYTHON_BIN="${PYTHON_BIN:-python}"
LLAMA_FACTORY_DIR="${LLAMA_FACTORY_DIR:-../LLaMA-Factory}"
CONDA_EXE="${CONDA_EXE:-$ROOT/scripts/conda_run_compat.sh}"
BASE_1B="${BASE_1B:-../models/Llama-3.2-1B}"
LORA_1B_TEMPLATE="${LORA_1B_TEMPLATE:-$LLAMA_FACTORY_DIR/examples/train_lora/llama3_lora_sft_probllm_1b_seed42.yaml}"
SEEDS=(${SEEDS:-42 2020 2021 2022 2023})
DATASETS="${DATASETS:-CiteULike ml-1m}"
MODEL_TEMPLATE="${MODEL_TEMPLATE:-llama3}"
METHOD="${METHOD:-probllm_llama32_1b}"
WARMUP_K="${WARMUP_K:-5}"
CAGA_K0="${CAGA_K0:-$WARMUP_K}"
RWFT_WEIGHTED="${RWFT_WEIGHTED:-1}"
RWFT_BETA="${RWFT_BETA:-0.8}"
CAGA_GAMMA="${CAGA_GAMMA:-1.0}"
EPOCHS="${EPOCHS:-100}"
WAIT_MODEL_SLEEP_SEC="${WAIT_MODEL_SLEEP_SEC:-60}"

wait_for_model() {
  while [[ ! -s "$BASE_1B/config.json" || ! -s "$BASE_1B/model.safetensors" ]]; do
    echo "Waiting for LLaMA-3.2-1B files in $BASE_1B"
    sleep "$WAIT_MODEL_SLEEP_SEC"
  done
}

dataset_key() {
  case "$1" in
    CiteULike) printf '%s\n' "CiteULike_item" ;;
    ml-1m) printf '%s\n' "ml-1m_item" ;;
    *) echo "Unsupported dataset: $1" >&2; return 1 ;;
  esac
}

run_seed_prediction() {
  local dataset="$1"
  local key="$2"
  local seed="$3"
  local experiment_name="${key}_llama32_1b_size_sensitivity"
  local seed_dir="$ROOT/experiments/multiseed/$experiment_name/seed_${seed}"
  local pred="$seed_dir/predicted_cold_item_interaction.csv"

  if [[ -s "$pred" ]]; then
    echo "SKIP LLaMA-1B prediction $key seed=$seed"
    return 0
  fi

  mkdir -p "$seed_dir"
  case "$dataset" in
    CiteULike)
      COLD_ROOT="$ROOT" \
      PYTHON_BIN="$PYTHON_BIN" \
      CONDA_EXE="$CONDA_EXE" \
      LLAMA_FACTORY_DIR="$LLAMA_FACTORY_DIR" \
      LLAMA_FACTORY_DATA_DIR="$LLAMA_FACTORY_DIR/data" \
      BASE_MODEL="$BASE_1B" \
      MODEL_TEMPLATE="$MODEL_TEMPLATE" \
      LORA_TEMPLATE="$LORA_1B_TEMPLATE" \
      EXPERIMENT_NAME="$experiment_name" \
      MODEL_TAG="llama32-1b-citeulike-size" \
      WARMUP_K="$WARMUP_K" \
      HF_INFER_BATCH_SIZE="${HF_INFER_BATCH_SIZE:-8}" \
      LLM_EMBED_BATCH_SIZE="${LLM_EMBED_BATCH_SIZE:-8}" \
      LLM_EMBED_TORCH_DTYPE="${LLM_EMBED_TORCH_DTYPE:-bfloat16}" \
      START_STEP=1 END_STEP=9 \
        bash "$ROOT/scripts/run_seed_citeulike_item.sh" "$seed"
      ;;
    ml-1m)
      COLD_ROOT="$ROOT" \
      PYTHON_BIN="$PYTHON_BIN" \
      CONDA_EXE="$CONDA_EXE" \
      LLAMA_FACTORY_DIR="$LLAMA_FACTORY_DIR" \
      LLAMA_FACTORY_DATA_DIR="$LLAMA_FACTORY_DIR/data" \
      BASE_MODEL="$BASE_1B" \
      MODEL_TEMPLATE="$MODEL_TEMPLATE" \
      LORA_TEMPLATE="$LORA_1B_TEMPLATE" \
      EXPERIMENT_NAME="$experiment_name" \
      MODEL_TAG="llama32-1b-ml1m-item-size" \
      WARMUP_K="$WARMUP_K" \
      HF_INFER_BATCH_SIZE="${HF_INFER_BATCH_SIZE:-8}" \
      LLM_EMBED_BATCH_SIZE="${LLM_EMBED_BATCH_SIZE:-8}" \
      LLM_EMBED_TORCH_DTYPE="${LLM_EMBED_TORCH_DTYPE:-bfloat16}" \
      START_STEP=1 END_STEP=9 \
        bash "$ROOT/scripts/run_seed_ml1m_item.sh" "$seed"
      ;;
    *)
      echo "Unsupported dataset: $dataset" >&2
      return 1
      ;;
  esac

  if [[ ! -s "$pred" ]]; then
    echo "LLaMA-1B prediction was not generated: $pred" >&2
    return 1
  fi
}

run_downstream() {
  local dataset="$1"
  local key="$2"
  local seed="$3"
  local experiment_name="${key}_llama32_1b_size_sensitivity"
  local pred="$ROOT/experiments/multiseed/$experiment_name/seed_${seed}/predicted_cold_item_interaction.csv"
  local final_metrics="$ROOT/experiments/revision_main/$key/lgn/$METHOD/seed_${seed}/final_metrics.json"

  if [[ -s "$final_metrics" ]]; then
    echo "SKIP downstream $key $METHOD seed=$seed"
    return 0
  fi
  [[ -s "$pred" ]] || { echo "Missing prediction for downstream: $pred" >&2; return 1; }

  RWFT_WEIGHTED="$RWFT_WEIGHTED" \
  RWFT_BETA="$RWFT_BETA" \
  CAGA_GAMMA="$CAGA_GAMMA" \
  CAGA_K0="$CAGA_K0" \
  WARMUP_K="$WARMUP_K" \
    bash "$ROOT/scripts/run_revision_finalupdate.sh" \
      "$dataset" item lgn "$METHOD" "$seed" "$pred" "$EPOCHS"
}

aggregate_dataset() {
  local key="$1"
  "$PYTHON_BIN" -u "$ROOT/scripts/aggregate_multiseed_results.py" \
    --experiment_root "$ROOT/experiments/revision_main/$key/lgn/$METHOD" \
    --method_name "ProbLLM LLaMA-3.2-1B LightGCN $key" \
    --seeds "${SEEDS[@]}"
}

wait_for_model
export PATH="$(dirname "$PYTHON_BIN"):$PATH"

for dataset in $DATASETS; do
  key="$(dataset_key "$dataset")"
  for seed in "${SEEDS[@]}"; do
    run_seed_prediction "$dataset" "$key" "$seed"
    run_downstream "$dataset" "$key" "$seed"
  done
  aggregate_dataset "$key"
done
