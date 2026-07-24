# ProbLLM: LLM-Based Interaction Simulation for Cold-Start Recommendation

This repository contains the core source code for the paper:

**Turning Cold Entities Warm: LLM-Based User and Item Interaction Simulation for Recommendations**

ProbLLM uses a large language model as an interaction simulator for cold-start and warm-up recommendation. The pipeline retrieves compact candidate pools, scores candidate user-item pairs with an LLM-based refiner, converts high-confidence scores into reliability-weighted pseudo-interactions, and updates downstream MF/LightGCN recommenders.

## Repository Layout

- `data/`: dataset split, conversion, prediction-check utilities, and data preparation notes.
- `FinalUpdate/`: downstream MF/LightGCN training and evaluation code.
- `configs/main_table_protocol.json`: the four setting-specific configurations used by the matched five-seed main tables.
- `scripts/`: core pipeline scripts for candidate construction, LLM scoring, pseudo-edge generation, diagnostics, and aggregation.
- `sft_data_generation.py`: builds LLM instruction-tuning examples.
- `extract_llm_embeddings.py`: extracts LLM-side representations.
- `train_llama_subtower.py`: trains the LLM embedding projection module.
- `generate_filtered_users_for_items.py`: builds filtered candidate users for cold items.

## Installation

Create a Python environment and install the lightweight Python dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The full LLM fine-tuning and inference stages require an external LLM stack such as LLaMA-Factory, plus access to the corresponding base checkpoints and LoRA adapters.

## LLM Setup

All LLM-facing scripts use Hugging Face Transformers paths. Prepare a base checkpoint in HF format, for example:

```bash
huggingface-cli login
huggingface-cli download meta-llama/Llama-2-7b-hf --local-dir models/Llama-2-7b-hf
```

The main LLaMA-2-7B setting uses `sft_data_generation.py` to build `data/<dataset>/train_sample.json`, then trains a LoRA adapter with an external SFT tool such as [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory). Register the generated JSON file in that tool's dataset config and use the LLaMA-2 instruction template. The paper runs use bf16, cutoff length 1024, LoRA rank 8, LoRA alpha 16, dropout 0, `lora_target=all`, learning rate `1e-4`, cosine scheduling, warmup ratio 0.1, per-device batch size 1, gradient accumulation 8, and 2 epochs.

After SFT, pass the base model and adapter paths directly:

```bash
python extract_llm_embeddings.py \
  --model_dir models/Llama-2-7b-hf \
  --adapter_model_path outputs/probllm_lora \
  --dataset CiteULike

python scripts/score_llm_pairwise.py \
  --model_name_or_path models/Llama-2-7b-hf \
  --adapter_name_or_path outputs/probllm_lora \
  --eval_json data/CiteULike/candidate_pairs.json \
  --save_name outputs/citeulike_predictions.jsonl \
  --prompt_template legacy
```

To score with a merged model instead of a separate adapter, first run `scripts/merge_lora_adapter.py`, then pass the merged directory as `--model_name_or_path`. Use `--prompt_template llama3` for LLaMA-3-style checkpoints and `--prompt_template qwen` for Qwen-style checkpoints.

## Data Preparation

The code expects benchmark dataset folders to be placed under `data/` following the experimental protocol.

See [`data/README.md`](data/README.md) for dataset sources, processed data characteristics, expected local file layout, and the split/conversion protocol. Raw datasets remain under their upstream licenses.

## Main-Table Reproduction

The generic scripts below are useful for development, but the reported main
tables use the setting-specific protocol in
[`configs/main_table_protocol.json`](configs/main_table_protocol.json).  It
fixes the five matched seeds (`42`, `2020`, `2021`, `2022`, `2023`), the
candidate and selection budgets, score readout and reliability encoding,
RWFT/CAGA values, CAGA target endpoint, warm-checkpoint update schedule, and
the MovieLens item-side score prior.  The protocol covers CiteULike item,
MovieLens item, MovieLens user, and Book-Crossing user cold-start settings.

For each seed, first construct the split using only training-graph
interactions, fit the seed-specific LoRA adapter, retrieve and score the
setting's candidate pool, and write the selected pseudo-edge CSV.  The CSV
must contain `user`, `item`, and `probability`, where `probability` is the
fixed reliability encoding specified in the protocol.  Then launch the
downstream update in a new isolated work directory:

```bash
bash scripts/run_main_table_finalupdate.sh \
  --setting citeulike_item \
  --seed 42 --model lgn \
  --edge-file /path/to/seed_42/pseudo_edges.csv \
  --init-checkpoint /path/to/seed_42/warm_checkpoint.pth.tar \
  --data-root /path/to/raw_data_root \
  --runroot /dev/shm/huangong/probllm_main_table/citeulike_lgn_seed42 \
  --output-root /path/to/durable_main_table_outputs
```

`--runroot` must be a new directory.  The launcher copies the code, requested
raw interaction file, pseudo-edge CSV, checkpoint, and (when applicable)
score prior into that directory before training, regenerates the split for the
supplied seed, and records the commands, hashes, and configuration in the
durable output directory.  For
`movielens_item`, additionally provide `--score-prior-file`; the protocol
requires the separate top-100 score prior and applies its fixed coefficient
only at ranking time.  Use `--dry-run` to inspect the selected parameters
without creating files.  The launcher defaults to `python3`; set
`PYTHON_BIN=/path/to/python` when the required environment uses a different
interpreter.

To aggregate a complete set of seed outputs and create the paper's
descriptive mean/SD rows and paired tests, declare the comparison row before
aggregation and run:

```bash
python3 scripts/aggregate_main_table_results.py \
  --method ProbLLM=/path/to/probllm_outputs/citeulike_item/lgn \
  --method BestBaseline=/path/to/baseline_outputs/citeulike_item/lgn \
  --target ProbLLM \
  --comparator BestBaseline \
  --output-dir /path/to/aggregates/citeulike_item_lgn
```

The script performs a two-sided paired t-test on the five matched seeds for
each split/metric cell.  The unadjusted cellwise p-values are the values used
for the paper's `*` and `**` markers (`p < 0.05` and `p < 0.01`,
respectively).  They are auxiliary cellwise tests, not a familywise
confirmatory analysis; no multiple-testing adjustment is applied.

The aggregator never chooses a comparison method from the observed target
results.  Use `--comparator NAME` when one comparison row is fixed for every
cell.  If the table protocol specifies different strongest comparison rows
for different cells, record that choice before running the significance tests
and pass `--comparator-map FILE`.  The JSON format is:

```json
{
  "overall": {
    "recall@20": "BaselineA",
    "ndcg@20": "BaselineA"
  },
  "strict_cold": {
    "recall@20": "BaselineB",
    "ndcg@20": "BaselineB"
  },
  "warmup": {
    "recall@20": "BaselineA",
    "ndcg@20": "BaselineB"
  }
}
```

Every method named by the fixed comparator or map must also be supplied with
`--method NAME=EXPERIMENT_ROOT`.  The script fails rather than silently
changing the test if any requested matched seed is missing.

## Typical Pipeline

1. Build seed-specific dataset splits:

```bash
python data/split.py --dataset CiteULike --warmup_k 5
python data/convert.py --dataset CiteULike --protocol warmup --warmup_k 5
```

2. Generate LLM instruction-tuning data:

```bash
python sft_data_generation.py --dataset CiteULike
```

3. Fine-tune or load the LLM simulator externally, then extract embeddings:

```bash
python extract_llm_embeddings.py --adapter_model_path /path/to/lora_or_model
```

4. Build candidate pools:

```bash
python scripts/build_embedding_item_topm_candidates.py --help
python scripts/build_text_neighbor_topm_candidates.py --help
python scripts/build_paper_style_llm_candidates.py --help
```

5. Score candidate pairs:

```bash
python scripts/score_llm_pairwise.py --help
```

6. Convert scores into pseudo-interactions:

```bash
python scripts/build_item_pseudo_from_predictions.py --help
python scripts/build_user_pseudo_from_predictions_blend.py --help
```

7. Run downstream recommender training/evaluation:

```bash
python FinalUpdate/main.py --help
```

8. Aggregate multi-seed metrics:

```bash
python scripts/aggregate_multiseed_results.py --help
python scripts/summarize_seed_outputs.py --help
```
