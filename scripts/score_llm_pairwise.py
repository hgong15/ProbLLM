import argparse
import json
import math
import os
import re
import time
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from peft import PeftModel
except ImportError:  # pragma: no cover - PEFT is only needed for adapter scoring.
    PeftModel = None


def count_jsonl(path):
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def pid_alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def acquire_output_lock(save_path, total):
    lock_path = Path(str(save_path) + ".lock")
    while True:
        existing = count_jsonl(save_path)
        if existing >= total:
            print(f"Existing prediction file is complete: {existing}/{total} in {save_path}")
            return None
        try:
            lock_path.mkdir()
            (lock_path / "pid").write_text(str(os.getpid()), encoding="utf-8")
            return lock_path
        except FileExistsError:
            pid_file = lock_path / "pid"
            pid_text = pid_file.read_text(encoding="utf-8").strip() if pid_file.exists() else ""
            if pid_text.isdigit() and not pid_alive(int(pid_text)):
                try:
                    pid_file.unlink(missing_ok=True)
                    lock_path.rmdir()
                    print(f"Removed stale output lock: {lock_path}")
                    continue
                except OSError:
                    pass
            print(f"Waiting for output lock: {lock_path}; current={existing}/{total}")
            time.sleep(60)


def release_output_lock(lock_path):
    if lock_path is None:
        return
    try:
        (lock_path / "pid").unlink(missing_ok=True)
        lock_path.rmdir()
    except OSError:
        pass


def _probllm_content(example):
    instruction = str(example.get("instruction", "")).strip()
    query = str(example.get("input", "")).strip()
    if query:
        return f"{instruction}\n\n{query}".strip()
    return instruction


def _llama3_prompt(content):
    return (
        "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
        f"{content}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    )


def _qwen_prompt(content):
    return (
        "<|im_start|>system\n"
        "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\n"
        f"{content}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def _apply_prompt_template(content, prompt_template):
    if prompt_template == "llama3":
        return _llama3_prompt(content)
    if prompt_template == "qwen":
        return _qwen_prompt(content)
    return content


def build_prompt(example, prompt_template="legacy"):
    content = _probllm_content(example)
    if prompt_template in {"llama3", "qwen"}:
        return _apply_prompt_template(content, prompt_template)
    return f"[INST] {content} [/INST] "


def label_token_ids(tokenizer, label):
    token_ids = tokenizer.encode(label, add_special_tokens=False)
    if not token_ids:
        raise ValueError(f"Could not tokenize label: {label!r}")
    return token_ids


def sequence_logprob(model, tokenizer, prompts, label_text, label_ids, device, max_length):
    full_texts = [prompt + label_text for prompt in prompts]
    encoded = tokenizer(
        full_texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}

    with torch.inference_mode():
        logits = model(**encoded).logits.float()
    log_probs = torch.log_softmax(logits[:, :-1, :], dim=-1)
    input_ids = encoded["input_ids"]

    scores = []
    label_len = len(label_ids)
    seq_len = input_ids.shape[1]
    full_lengths = encoded["attention_mask"].sum(dim=1).tolist()
    for row_idx, full_len in enumerate(full_lengths):
        start = seq_len - label_len - 1
        end = start + label_len
        if full_len <= label_len or start < 0 or end > log_probs.shape[1]:
            scores.append(float("-inf"))
            continue
        target = input_ids[row_idx, start + 1 : end + 1]
        token_scores = log_probs[row_idx, start:end].gather(1, target.unsqueeze(1)).squeeze(1)
        scores.append(token_scores.sum().item())
    return scores


def score_batch(model, tokenizer, prompts, positive_label, positive_ids, negative_label, negative_ids, device, max_length):
    pos_scores = sequence_logprob(model, tokenizer, prompts, positive_label, positive_ids, device, max_length)
    neg_scores = sequence_logprob(model, tokenizer, prompts, negative_label, negative_ids, device, max_length)
    probs = []
    for pos, neg in zip(pos_scores, neg_scores):
        if math.isinf(pos) and math.isinf(neg):
            probs.append(0.0)
            continue
        pair = torch.tensor([neg, pos], dtype=torch.float32)
        probs.append(torch.softmax(pair, dim=0)[1].item())
    return probs


def first_content_token(tokenizer, text):
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    for token_id in token_ids:
        if tokenizer.decode([token_id]).strip():
            return token_id
    raise ValueError(f"Could not find content token for label: {text!r}")


def score_batch_next_token(model, tokenizer, prompts, positive_id, negative_id, device, max_length):
    encoded = tokenizer(
        prompts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}
    with torch.inference_mode():
        logits = model(**encoded).logits.float()
    # Works for both left and right padding. With left padding, attention_mask.sum()-1
    # points into the pad prefix rather than the final prompt token.
    positions = torch.arange(encoded["attention_mask"].shape[1], device=device).unsqueeze(0)
    last_nonpad = (encoded["attention_mask"].long() * positions).max(dim=1).values
    next_logits = logits[torch.arange(logits.shape[0], device=device), last_nonpad]
    label_logits = torch.stack([next_logits[:, negative_id], next_logits[:, positive_id]], dim=1)
    return torch.softmax(label_logits, dim=1)[:, 1].detach().cpu().tolist()


def parse_probability(text, fallback=0.0):
    match = re.search(r"(?<![\d.])(?:0\.\d+|1\.0+)(?!\d)", text)
    if not match:
        stripped = text.strip()
        if stripped not in {"0", "1"}:
            return fallback
        value = float(stripped)
    else:
        value = float(match.group(0))
    if value < 0.0 or value > 1.0 or math.isnan(value):
        return fallback
    return value


def score_batch_generate(model, tokenizer, prompts, device, max_length, max_new_tokens, fallback):
    encoded = tokenizer(
        prompts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}
    input_len = encoded["input_ids"].shape[1]
    with torch.inference_mode():
        outputs = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    decoded = tokenizer.batch_decode(outputs[:, input_len:], skip_special_tokens=True)
    return [parse_probability(text, fallback=fallback) for text in decoded], decoded


def main():
    parser = argparse.ArgumentParser(description="ProbLLM candidate-pair scoring.")
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--eval_json", required=True)
    parser.add_argument("--save_name", required=True)
    parser.add_argument("--mode", default="probllm", choices=["probllm"])
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--no_resume", action="store_true")
    parser.add_argument("--score_method", default="next_token", choices=["next_token", "sequence", "generate"])
    parser.add_argument("--prompt_template", default="legacy", choices=["legacy", "llama3", "qwen"])
    parser.add_argument("--adapter_name_or_path", default=None)
    parser.add_argument("--tokenizer_name_or_path", default=None)
    parser.add_argument("--max_new_tokens", type=int, default=8)
    parser.add_argument("--invalid_probability", type=float, default=0.0)
    parser.add_argument("--torch_dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    args = parser.parse_args()

    dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[args.torch_dtype]

    examples = json.loads(Path(args.eval_json).read_text(encoding="utf-8"))
    if args.max_samples is not None:
        examples = examples[: args.max_samples]

    save_path = Path(args.save_name)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    total = len(examples)
    lock_path = acquire_output_lock(save_path, total)
    if lock_path is None:
        return
    try:
        tokenizer_source = args.tokenizer_name_or_path or args.model_name_or_path
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            torch_dtype=dtype,
            device_map="auto",
            trust_remote_code=True,
        )
        if args.adapter_name_or_path:
            if PeftModel is None:
                raise RuntimeError("PEFT is required for --adapter_name_or_path but is not installed.")
            model = PeftModel.from_pretrained(model, args.adapter_name_or_path)
        model.eval()
        device = next(model.parameters()).device

        positive_label = "1"
        negative_label = "0"
        positive_ids = label_token_ids(tokenizer, positive_label)
        negative_ids = label_token_ids(tokenizer, negative_label)
        positive_next_id = first_content_token(tokenizer, positive_label)
        negative_next_id = first_content_token(tokenizer, negative_label)

        _run_scoring(args, model, tokenizer, examples, save_path, total, positive_label, positive_ids, negative_label, negative_ids, positive_next_id, negative_next_id, device)
    finally:
        release_output_lock(lock_path)


def _run_scoring(args, model, tokenizer, examples, save_path, total, positive_label, positive_ids, negative_label, negative_ids, positive_next_id, negative_next_id, device):
    start_offset = 0
    file_mode = "w"
    if save_path.exists() and not args.no_resume:
        start_offset = count_jsonl(save_path)
        start_offset = min(start_offset, total)
        file_mode = "a"
        if start_offset:
            print(f"Resuming from {start_offset}/{total} scored pairs in {save_path}")

    with save_path.open(file_mode, encoding="utf-8") as f:
        for start in tqdm(range(start_offset, total, args.batch_size), desc=f"Scoring {args.mode}"):
            batch = examples[start : start + args.batch_size]
            prompts = [build_prompt(example, args.prompt_template) for example in batch]
            if args.score_method == "next_token":
                probs = score_batch_next_token(
                    model, tokenizer, prompts, positive_next_id, negative_next_id, device, args.max_length
                )
                generations = [""] * len(probs)
            elif args.score_method == "generate":
                probs, generations = score_batch_generate(
                    model,
                    tokenizer,
                    prompts,
                    device,
                    args.max_length,
                    args.max_new_tokens,
                    args.invalid_probability,
                )
            else:
                probs = score_batch(
                    model,
                    tokenizer,
                    prompts,
                    positive_label,
                    positive_ids,
                    negative_label,
                    negative_ids,
                    device,
                    args.max_length,
                )
                generations = [""] * len(probs)
            for example, prompt, prob, generation in zip(batch, prompts, probs, generations):
                if math.isnan(prob):
                    prob = 0.0
                row = {
                    "prompt": prompt,
                    "predict": f"{prob:.6f}",
                    "label": example.get("output", ""),
                    "user_id": example.get("user_id"),
                    "item_id": example.get("item_id"),
                    "entity_type": example.get("entity_type"),
                }
                if generation:
                    row["generation"] = generation
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()

    print(f"Saved {total} scored pairs to {save_path}")


if __name__ == "__main__":
    main()
