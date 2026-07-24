#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-.}"
PYTHON_BIN="${PYTHON_BIN:-python}"
SEED="${SEED:-42}"
BATCH_SIZE="${BATCH_SIZE:-16}"
MAX_LENGTH="${MAX_LENGTH:-2048}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-8}"
TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"
EPOCHS="${EPOCHS:-100}"
RWFT_WEIGHTED="${RWFT_WEIGHTED:-1}"
RWFT_BETA="${RWFT_BETA:-0.8}"
CAGA_GAMMA="${CAGA_GAMMA:-1.0}"
CAGA_K0="${CAGA_K0:-5}"
WARMUP_K="${WARMUP_K:-5}"
DATASETS="${DATASETS:-CiteULike ml-1m}"
VARIANTS="${VARIANTS:-default concise numeric_only preference_match}"
OUT_ROOT="${OUT_ROOT:-$ROOT/experiments/revision_diagnostics/prompt_sensitivity}"
LOCK_ROOT="${LOCK_ROOT:-$ROOT/experiments/.locks}"

mkdir -p "$OUT_ROOT" "$LOCK_ROOT"

dataset_key() {
  case "$1" in
    CiteULike) printf '%s\n' "CiteULike_item" ;;
    ml-1m) printf '%s\n' "ml-1m_item" ;;
    *) echo "Unsupported dataset: $1" >&2; return 1 ;;
  esac
}

domain_for_dataset() {
  case "$1" in
    CiteULike) printf '%s\n' "paper" ;;
    ml-1m) printf '%s\n' "movie" ;;
    *) echo "Unsupported dataset: $1" >&2; return 1 ;;
  esac
}

top20_for_dataset() {
  case "$1" in
    CiteULike)
      printf '%s\n' "$ROOT/experiments/multiseed/CiteULike_item/seed_${SEED}/top20.csv"
      ;;
    ml-1m)
      printf '%s\n' "$ROOT/experiments/multiseed/ml-1m_item_sample2000_rerun20260624/seed_${SEED}/top20.csv"
      ;;
    *) echo "Unsupported dataset: $1" >&2; return 1 ;;
  esac
}

model_for_dataset() {
  case "$1" in
    CiteULike)
      printf '%s\n' "$ROOT/weight/llama2-7b-citeulike-sample2000_rerun20260624_seed${SEED}_merged"
      ;;
    ml-1m)
      printf '%s\n' "$ROOT/weight/llama2-7b-ml1m-item-sample2000-rerun20260624_seed${SEED}_merged"
      ;;
    *) echo "Unsupported dataset: $1" >&2; return 1 ;;
  esac
}

count_jsonl() {
  local path="$1"
  if [[ -s "$path" ]]; then
    grep -cve '^[[:space:]]*$' "$path" || true
  else
    printf '0\n'
  fi
}

expected_examples() {
  "$PYTHON_BIN" - "$1" <<'PY'
import json
import sys
from pathlib import Path

print(len(json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))))
PY
}

run_finalupdate_locked() {
  local dataset="$1"
  local key="$2"
  local method="$3"
  local csv="$4"
  local metrics="$ROOT/experiments/revision_main/$key/lgn/$method/seed_${SEED}/final_metrics.json"
  local lock="$LOCK_ROOT/${key}.finalupdate.lock"

  if [[ -s "$metrics" ]]; then
    echo "SKIP downstream $method: $metrics exists"
    return 0
  fi

  while ! mkdir "$lock" 2>/dev/null; do
    echo "Waiting for dataset FinalUpdate lock: $lock"
    sleep 30
  done
  (
    export RWFT_WEIGHTED RWFT_BETA CAGA_GAMMA CAGA_K0 WARMUP_K
    bash "$ROOT/scripts/run_revision_finalupdate.sh" \
      "$dataset" item lgn "$method" "$SEED" "$csv" "$EPOCHS"
  )
  rmdir "$lock"
}

for dataset in $DATASETS; do
  key="$(dataset_key "$dataset")"
  domain="$(domain_for_dataset "$dataset")"
  top20="$(top20_for_dataset "$dataset")"
  model_path="$(model_for_dataset "$dataset")"

  [[ -s "$top20" ]] || { echo "Missing top20: $top20" >&2; exit 1; }
  [[ -d "$model_path" ]] || { echo "Missing merged model: $model_path" >&2; exit 1; }

  for variant in $VARIANTS; do
    out_dir="$OUT_ROOT/$key/seed_${SEED}/$variant"
    mkdir -p "$out_dir"
    eval_json="$out_dir/${dataset}_eval_${variant}_seed${SEED}.json"
    pred_jsonl="$out_dir/generated_predictions_${variant}.jsonl"
    sim_csv="$out_dir/predicted_cold_item_interaction_${variant}.csv"
    method="probllm_prompt_${variant}_seed${SEED}"

    cp -f "$top20" "$out_dir/top20.csv"

    if [[ ! -s "$eval_json" ]]; then
      "$PYTHON_BIN" -u "$ROOT/scripts/build_citeulike_eval_from_top20.py" \
        --data_dir "$ROOT/data/$dataset" \
        --top20_csv "$out_dir/top20.csv" \
        --domain "$domain" \
        --prompt_variant "$variant" \
        --output_json "$eval_json"
    fi

    expected="$(expected_examples "$eval_json")"
    current="$(count_jsonl "$pred_jsonl")"
    if [[ "$current" -lt "$expected" ]]; then
      "$PYTHON_BIN" -u "$ROOT/scripts/score_llm_pairwise.py" \
        --model_name_or_path "$model_path" \
        --eval_json "$eval_json" \
        --save_name "$pred_jsonl" \
        --mode probllm \
        --batch_size "$BATCH_SIZE" \
        --max_length "$MAX_LENGTH" \
        --score_method generate \
        --max_new_tokens "$MAX_NEW_TOKENS" \
        --torch_dtype "$TORCH_DTYPE"
    else
      echo "SKIP scoring $key $variant: $current/$expected"
    fi

    "$PYTHON_BIN" -u "$ROOT/data/CiteULike/check.py" \
      --jsonl_path "$pred_jsonl" \
      --top20_csv_path "$out_dir/top20.csv" \
      --output_csv_path "$sim_csv"

    run_finalupdate_locked "$dataset" "$key" "$method" "$sim_csv"
  done
done
