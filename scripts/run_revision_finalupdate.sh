#!/usr/bin/env bash
set -euo pipefail

DATASET="${1:?Usage: bash scripts/run_revision_finalupdate.sh <dataset> <cold_object> <model> <method> <seed> <extended_csv> [epochs]}"
COLD_OBJECT="${2:?Usage: bash scripts/run_revision_finalupdate.sh <dataset> <cold_object> <model> <method> <seed> <extended_csv> [epochs]}"
MODEL="${3:?Usage: bash scripts/run_revision_finalupdate.sh <dataset> <cold_object> <model> <method> <seed> <extended_csv> [epochs]}"
METHOD="${4:?Usage: bash scripts/run_revision_finalupdate.sh <dataset> <cold_object> <model> <method> <seed> <extended_csv> [epochs]}"
SEED="${5:?Usage: bash scripts/run_revision_finalupdate.sh <dataset> <cold_object> <model> <method> <seed> <extended_csv> [epochs]}"
EXTENDED_CSV="${6:?Usage: bash scripts/run_revision_finalupdate.sh <dataset> <cold_object> <model> <method> <seed> <extended_csv> [epochs]}"
EPOCHS="${7:-100}"

ROOT="${ROOT:-.}"
PYTHON_BIN="${PYTHON_BIN:-python}"
PROBLLM_ENV="${PROBLLM_ENV:-probllm}"
CONDA_EXE="${CONDA_EXE:-conda}"
WARMUP_K="${WARMUP_K:-5}"
NVIDIA_DRIVER_LIB_DIR="${NVIDIA_DRIVER_LIB_DIR:-/usr/local/nvidia/lib64}"
RWFT_WEIGHTED="${RWFT_WEIGHTED:-0}"
RWFT_BETA="${RWFT_BETA:-0.8}"
CAGA_GAMMA="${CAGA_GAMMA:-1.0}"
CAGA_K0="${CAGA_K0:-$WARMUP_K}"
SIM_PROB_COLUMN="${SIM_PROB_COLUMN:-probability}"
TEST_BATCH="${TEST_BATCH:-512}"
BPR_BATCH="${BPR_BATCH:-128}"

DATA_DIR="$ROOT/data/$DATASET"
EXPERIMENT_ROOT="$ROOT/experiments/revision_main/${DATASET}_${COLD_OBJECT}/${MODEL}/${METHOD}"
SEED_DIR="$EXPERIMENT_ROOT/seed_${SEED}"
LOG_FILE="$SEED_DIR/finalupdate.log"
COMMANDS_FILE="$SEED_DIR/commands.txt"
RUN_ID="revision_${DATASET}_${COLD_OBJECT}_${MODEL}_${METHOD}_seed${SEED}"
GRAPH_CACHE="fin_s_pre_adj_mat_${RUN_ID}.npz"

if [[ ! -d "$DATA_DIR" ]]; then
  echo "Dataset directory not found: $DATA_DIR" >&2
  exit 1
fi

if [[ ! -f "$EXTENDED_CSV" ]]; then
  echo "Extended interaction CSV not found: $EXTENDED_CSV" >&2
  exit 1
fi
EXTENDED_CSV="$(readlink -f "$EXTENDED_CSV")"

if [[ "$MODEL" != "mf" && "$MODEL" != "lgn" ]]; then
  echo "MODEL must be mf or lgn, got: $MODEL" >&2
  exit 1
fi

mkdir -p "$SEED_DIR"
: > "$LOG_FILE"
: > "$COMMANDS_FILE"

if [[ -d "$NVIDIA_DRIVER_LIB_DIR" ]]; then
  export LD_LIBRARY_PATH="$NVIDIA_DRIVER_LIB_DIR:${LD_LIBRARY_PATH:-}"
fi

record_cmd() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*" >> "$COMMANDS_FILE"
}

run_cmd() {
  local env_name="$1"
  local workdir="$2"
  shift 2
  local -a cmd=("$@")
  if [[ "${cmd[0]}" == "python" ]]; then
    cmd[0]="$PYTHON_BIN"
  fi
  record_cmd "cd $workdir && ${cmd[*]}"
  printf '\n[%s] RUN (%s @ %s): %s\n' "$(date '+%F %T')" "$env_name" "$workdir" "${cmd[*]}" | tee -a "$LOG_FILE"
  (
    cd "$workdir"
    "${cmd[@]}"
  ) 2>&1 | tee -a "$LOG_FILE"
}

save_split_artifacts() {
  for name in \
    split_meta.json \
    convert_dict.pkl \
    warm_emb.csv \
    warm_train.csv \
    warm_emb_original.csv \
    warm_train_original.csv \
    warm_val.csv \
    warm_test.csv \
    warmup_support.csv \
    warmup_val.csv \
    warmup_test.csv \
    cold_${COLD_OBJECT}.csv \
    cold_${COLD_OBJECT}_val.csv \
    cold_${COLD_OBJECT}_test.csv \
    overall_val.csv \
    overall_test.csv
  do
    if [[ -f "$DATA_DIR/$name" ]]; then
      cp -f "$DATA_DIR/$name" "$SEED_DIR/$name"
    fi
  done
  cp -f "$EXTENDED_CSV" "$SEED_DIR/extended_interactions.csv"
}

printf '[%s] Starting revision FinalUpdate: dataset=%s cold_object=%s model=%s method=%s seed=%s epochs=%s\n' \
  "$(date '+%F %T')" "$DATASET" "$COLD_OBJECT" "$MODEL" "$METHOD" "$SEED" "$EPOCHS" | tee -a "$LOG_FILE"

run_cmd "$PROBLLM_ENV" "$ROOT/data" \
  python -u split.py \
    --dataset "$DATASET" \
    --seed "$SEED" \
    --warmup_k "$WARMUP_K" \
    --cold_object "$COLD_OBJECT"

run_cmd "$PROBLLM_ENV" "$ROOT/data" \
  python -u convert.py \
    --dataset "$DATASET" \
    --seed "$SEED" \
    --protocol warmup \
    --warmup_k "$WARMUP_K" \
    --cold_object "$COLD_OBJECT"

record_cmd "rm -f $DATA_DIR/$GRAPH_CACHE"
rm -f "$DATA_DIR/$GRAPH_CACHE"

run_cmd "$PROBLLM_ENV" "$ROOT/FinalUpdate" \
  python -u main.py \
    --dataset "$DATASET" \
    --seed "$SEED" \
    --model "$MODEL" \
    --load 0 \
    --epochs "$EPOCHS" \
    --bpr_batch "$BPR_BATCH" \
    --testbatch "$TEST_BATCH" \
    --file_name "$RUN_ID" \
    --extended_file "$EXTENDED_CSV" \
    --graph_cache "$GRAPH_CACHE" \
    --rwft_weighted "$RWFT_WEIGHTED" \
    --rwft_beta "$RWFT_BETA" \
    --caga_gamma "$CAGA_GAMMA" \
    --caga_k0 "$CAGA_K0" \
    --sim_prob_column "$SIM_PROB_COLUMN"

run_cmd "$PROBLLM_ENV" "$ROOT" \
  python -u scripts/summarize_seed_outputs.py \
    --seed "$SEED" \
    --data_dir "$DATA_DIR" \
    --seed_dir "$SEED_DIR" \
    --log_file "$LOG_FILE"

save_split_artifacts

printf '[%s] Finished revision FinalUpdate: %s\n' "$(date '+%F %T')" "$SEED_DIR" | tee -a "$LOG_FILE"
