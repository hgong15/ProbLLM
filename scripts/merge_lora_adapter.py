#!/usr/bin/env python
import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--adapter_model", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(model, args.adapter_model)
    model = model.merge_and_unload()
    model.save_pretrained(args.output_dir, safe_serialization=True)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Merged LoRA adapter saved to {args.output_dir}")


if __name__ == "__main__":
    main()
