#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-.}"
PYTHON_BIN="${PYTHON_BIN:-python}"
OUT_ROOT="${OUT_ROOT:-results/revision_diagnostics/no_context_probe}"
BATCH_SIZE="${BATCH_SIZE:-16}"
MAX_LENGTH="${MAX_LENGTH:-1024}"
TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"

run_one() {
  local dataset="$1"
  local out_dir="$2"
  local model="$3"
  local pred="$out_dir/generated_predictions.jsonl"
  local eval_json="$out_dir/eval_no_context.json"
  local summary="$out_dir/summary.json"

  [[ -s "$eval_json" ]] || { echo "Missing eval json: $eval_json" >&2; exit 1; }
  [[ -d "$model" ]] || { echo "Missing model: $model" >&2; exit 1; }

  "$PYTHON_BIN" -u "$ROOT/scripts/score_llm_pairwise.py" \
    --model_name_or_path "$model" \
    --eval_json "$eval_json" \
    --save_name "$pred" \
    --mode probllm \
    --batch_size "$BATCH_SIZE" \
    --max_length "$MAX_LENGTH" \
    --score_method generate \
    --max_new_tokens 8 \
    --torch_dtype "$TORCH_DTYPE"

  "$PYTHON_BIN" -u "$ROOT/scripts/build_no_context_memorization_probe.py" \
    --dataset "$dataset" \
    --data_dir "$ROOT/data/$dataset" \
    --output_json "$eval_json" \
    --output_pairs_csv "$out_dir/pairs.csv" \
    --prediction_jsonl "$pred" \
    --summary_json "$summary" \
    --max_per_split 300
}

run_one \
  ml-1m \
  "$OUT_ROOT/ml1m_item_seed42" \
  "$ROOT/weight/llama2-7b-ml1m-item-sample2000-rerun20260624_seed42_merged"

run_one \
  CiteULike \
  "$OUT_ROOT/citeulike_item_seed42" \
  "$ROOT/weight/llama2-7b-citeulike-sample2000_rerun20260624_seed42_merged"
