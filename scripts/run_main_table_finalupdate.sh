#!/usr/bin/env bash
# Run one canonical main-table downstream update in an isolated work directory.
# Candidate retrieval, LLM scoring, and pseudo-edge construction must be run
# first.  This launcher consumes their seed-specific pseudo-edge CSV, exactly
# records its inputs, and applies the downstream protocol in
# configs/main_table_protocol.json.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_main_table_finalupdate.sh \
    --setting <citeulike_item|movielens_item|movielens_user|bookcrossing_user> \
    --seed <seed> --model <mf|lgn> --edge-file <pseudo_edges.csv> \
    --init-checkpoint <warm_checkpoint.pth.tar> --data-root <raw_data_root> \
    --runroot <empty_isolated_directory> --output-root <durable_output_root> \
    [--score-prior-file <prior.csv>] [--config <protocol.json>] [--dry-run]

data-root must contain <dataset>/<dataset>.csv, where <dataset> is the value
in the selected protocol setting.  For MovieLens item-side runs,
--score-prior-file is mandatory.  The launcher never writes experiment output
inside the Git repository.
EOF
}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="$REPO_ROOT/configs/main_table_protocol.json"
PYTHON_BIN="${PYTHON_BIN:-python3}"
SETTING=""
SEED=""
MODEL=""
EDGE_FILE=""
INIT_CHECKPOINT=""
DATA_ROOT=""
RUNROOT=""
OUTPUT_ROOT=""
SCORE_PRIOR_FILE=""
DRY_RUN=0

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --setting) SETTING="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --edge-file) EDGE_FILE="$2"; shift 2 ;;
    --init-checkpoint) INIT_CHECKPOINT="$2"; shift 2 ;;
    --data-root) DATA_ROOT="$2"; shift 2 ;;
    --runroot) RUNROOT="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --score-prior-file) SCORE_PRIOR_FILE="$2"; shift 2 ;;
    --config) CONFIG="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

for required in SETTING SEED MODEL EDGE_FILE INIT_CHECKPOINT DATA_ROOT RUNROOT OUTPUT_ROOT; do
  [[ -n "${!required}" ]] || { echo "Missing --${required,,}" >&2; usage >&2; exit 2; }
done
[[ "$MODEL" == "mf" || "$MODEL" == "lgn" ]] || { echo "--model must be mf or lgn" >&2; exit 2; }
[[ -f "$CONFIG" ]] || { echo "Protocol config not found: $CONFIG" >&2; exit 2; }

readarray -t protocol < <("$PYTHON_BIN" - "$CONFIG" "$SETTING" <<'PY'
import json
import sys

config_path, setting_name = sys.argv[1:]
with open(config_path, encoding="utf-8") as handle:
    config = json.load(handle)
try:
    setting = config["settings"][setting_name]
except KeyError as exc:
    choices = ", ".join(sorted(config.get("settings", {})))
    raise SystemExit(f"Unknown setting {setting_name!r}; choose one of: {choices}") from exc
shared = config["shared"]
prior = setting.get("score_prior", {})
fields = [
    setting["dataset"], setting["cold_object"], str(shared["warmup_k"]),
    str(shared["lightgcn_layers"]), str(shared["l2_decay"]),
    str(shared["evaluation_batch_size"]), str(setting["rwft_beta"]),
    str(setting["caga_gamma"]), setting["caga_target_object"],
    setting["evaluation_candidate_mode"], str(setting["final_epochs"]),
    str(setting["final_learning_rate"]), str(setting["final_bpr_batch"]),
    "1" if setting["target_only_update"] else "0",
    "1" if prior.get("required", False) else "0",
    str(prior.get("coefficient", 0.0)),
]
print("\n".join(fields))
PY
)
[[ "${#protocol[@]}" -eq 16 ]] || { echo "Could not read setting $SETTING from $CONFIG" >&2; exit 2; }

DATASET="${protocol[0]}"
COLD_OBJECT="${protocol[1]}"
WARMUP_K="${protocol[2]}"
LAYER="${protocol[3]}"
DECAY="${protocol[4]}"
TEST_BATCH="${protocol[5]}"
RWFT_BETA="${protocol[6]}"
CAGA_GAMMA="${protocol[7]}"
CAGA_TARGET="${protocol[8]}"
EVAL_MODE="${protocol[9]}"
EPOCHS="${protocol[10]}"
LR="${protocol[11]}"
BPR_BATCH="${protocol[12]}"
TARGET_ONLY="${protocol[13]}"
PRIOR_REQUIRED="${protocol[14]}"
PRIOR_ALPHA="${protocol[15]}"

if [[ "$PRIOR_REQUIRED" == "1" && -z "$SCORE_PRIOR_FILE" ]]; then
  echo "The $SETTING protocol requires --score-prior-file." >&2
  exit 2
fi

if [[ "$DRY_RUN" == "0" ]]; then
  for path in "$EDGE_FILE" "$INIT_CHECKPOINT" "$DATA_ROOT/$DATASET/$DATASET.csv"; do
    [[ -s "$path" ]] || { echo "Required input missing or empty: $path" >&2; exit 1; }
  done
  if [[ -n "$SCORE_PRIOR_FILE" ]]; then
    [[ -s "$SCORE_PRIOR_FILE" ]] || { echo "Score-prior file missing or empty: $SCORE_PRIOR_FILE" >&2; exit 1; }
  fi
  [[ ! -e "$RUNROOT" ]] || { echo "--runroot must not already exist: $RUNROOT" >&2; exit 1; }
fi

RUN_ID="main_table_${SETTING}_${MODEL}_seed${SEED}"
SEED_DIR="$OUTPUT_ROOT/$SETTING/$MODEL/seed_${SEED}"
GRAPH_CACHE="fin_s_pre_adj_mat_${RUN_ID}.npz"

print_plan() {
  cat <<EOF
setting=$SETTING dataset=$DATASET cold_object=$COLD_OBJECT seed=$SEED model=$MODEL
rwft_beta=$RWFT_BETA caga_gamma=$CAGA_GAMMA caga_target=$CAGA_TARGET
epochs=$EPOCHS lr=$LR bpr_batch=$BPR_BATCH evaluation_mode=$EVAL_MODE
runroot=$RUNROOT output=$SEED_DIR
EOF
}

if [[ "$DRY_RUN" == "1" ]]; then
  print_plan
  exit 0
fi

mkdir -p "$RUNROOT" "$SEED_DIR"
rsync -a --exclude '.git' --exclude 'data' --exclude 'experiments' --exclude 'results' \
  --exclude 'logs' --exclude 'weight' --exclude 'weights' --exclude 'checkpoints' \
  --exclude '__pycache__' "$REPO_ROOT/" "$RUNROOT/"
mkdir -p "$RUNROOT/data/$DATASET" "$RUNROOT/code/checkpoints" "$RUNROOT/inputs"
cp "$REPO_ROOT/data/split.py" "$REPO_ROOT/data/convert.py" "$RUNROOT/data/"
cp "$DATA_ROOT/$DATASET/$DATASET.csv" "$RUNROOT/data/$DATASET/$DATASET.csv"

TARGET_CHECKPOINT="$RUNROOT/code/checkpoints/${MODEL}-${RUN_ID}.pth.tar"
INPUT_EDGE="$RUNROOT/inputs/pseudo_edges.csv"
INPUT_SCORE_PRIOR=""
cp "$INIT_CHECKPOINT" "$TARGET_CHECKPOINT"
cp "$EDGE_FILE" "$INPUT_EDGE"
if [[ -n "$SCORE_PRIOR_FILE" ]]; then
  INPUT_SCORE_PRIOR="$RUNROOT/inputs/score_prior.csv"
  cp "$SCORE_PRIOR_FILE" "$INPUT_SCORE_PRIOR"
fi
LOG_FILE="$SEED_DIR/finalupdate.log"
COMMAND_FILE="$SEED_DIR/commands.txt"
printf '' > "$LOG_FILE"
printf '' > "$COMMAND_FILE"

record_and_run() {
  local workdir="$1"
  shift
  printf '[%s] cd %q &&' "$(date -Is)" "$workdir" >> "$COMMAND_FILE"
  printf ' %q' "$@" >> "$COMMAND_FILE"
  printf '\n' >> "$COMMAND_FILE"
  (cd "$workdir" && "$@") 2>&1 | tee -a "$LOG_FILE"
}

record_and_run "$RUNROOT/data" "$PYTHON_BIN" -u split.py \
  --dataset "$DATASET" --seed "$SEED" --warmup_k "$WARMUP_K" --cold_object "$COLD_OBJECT"
record_and_run "$RUNROOT/data" "$PYTHON_BIN" -u convert.py \
  --dataset "$DATASET" --seed "$SEED" --protocol warmup --warmup_k "$WARMUP_K" --cold_object "$COLD_OBJECT"
rm -f "$RUNROOT/data/$DATASET/$GRAPH_CACHE"

export ROOT="$RUNROOT"
export PROBLLM_EVAL_CANDIDATE_MODE="$EVAL_MODE"
export FINALUPDATE_VALID_EVERY=5
export FINALUPDATE_BEST_METRIC=overall_ndcg@20
if [[ "$TARGET_ONLY" == "1" ]]; then
  export FINALUPDATE_TARGET_ONLY_UPDATE=1
  export FINALUPDATE_TARGET_ONLY_OBJECT="$COLD_OBJECT"
  export FINALUPDATE_TARGET_ONLY_FREEZE_ITEMS=1
  export FINALUPDATE_TARGET_ONLY_FREEZE_USERS=1
fi
if [[ -n "$SCORE_PRIOR_FILE" ]]; then
  export PROBLLM_SCORE_PRIOR_FILE="$INPUT_SCORE_PRIOR"
  export PROBLLM_SCORE_PRIOR_ALPHA="$PRIOR_ALPHA"
fi

record_and_run "$RUNROOT/FinalUpdate" "$PYTHON_BIN" -u main_best.py \
  --dataset "$DATASET" --seed "$SEED" --model "$MODEL" --load 1 \
  --epochs "$EPOCHS" --layer "$LAYER" --lr "$LR" --decay "$DECAY" \
  --bpr_batch "$BPR_BATCH" --testbatch "$TEST_BATCH" --file_name "$RUN_ID" \
  --extended_file "$INPUT_EDGE" --graph_cache "$GRAPH_CACHE" \
  --rwft_weighted 1 --rwft_beta "$RWFT_BETA" --caga_gamma "$CAGA_GAMMA" \
  --caga_k0 "$WARMUP_K" --caga_target_object "$CAGA_TARGET" \
  --sim_prob_column probability

record_and_run "$RUNROOT" "$PYTHON_BIN" -u scripts/summarize_seed_outputs.py \
  --seed "$SEED" --data_dir "$RUNROOT/data/$DATASET" --seed_dir "$SEED_DIR" --log_file "$LOG_FILE"

cp "$INPUT_EDGE" "$SEED_DIR/extended_interactions.csv"
if [[ -n "$INPUT_SCORE_PRIOR" ]]; then cp "$INPUT_SCORE_PRIOR" "$SEED_DIR/score_prior.csv"; fi
hash_inputs=("$EDGE_FILE" "$INIT_CHECKPOINT" "$TARGET_CHECKPOINT")
if [[ -n "$SCORE_PRIOR_FILE" ]]; then hash_inputs+=("$SCORE_PRIOR_FILE"); fi
sha256sum "${hash_inputs[@]}" > "$SEED_DIR/inputs_outputs.sha256"
"$PYTHON_BIN" - "$CONFIG" "$SETTING" "$SEED" "$MODEL" "$RUNROOT" "$SEED_DIR" "$EDGE_FILE" "$INIT_CHECKPOINT" "$SCORE_PRIOR_FILE" "$INPUT_EDGE" "$INPUT_SCORE_PRIOR" <<'PY' > "$SEED_DIR/run_manifest.json"
import json
import sys

config_path, setting, seed, model, runroot, output, edge, checkpoint, prior, isolated_edge, isolated_prior = sys.argv[1:]
with open(config_path, encoding="utf-8") as handle:
    config = json.load(handle)
json.dump(
    {
        "protocol": config["protocol_name"],
        "setting": setting,
        "seed": int(seed),
        "model": model,
        "runroot": runroot,
        "output": output,
        "pseudo_edge_file": edge,
        "isolated_pseudo_edge_file": isolated_edge,
        "warm_checkpoint": checkpoint,
        "score_prior_file": prior or None,
        "isolated_score_prior_file": isolated_prior or None,
        "setting_config": config["settings"][setting],
        "shared_config": config["shared"],
    },
    sys.stdout,
    indent=2,
)
print()
PY
print_plan
