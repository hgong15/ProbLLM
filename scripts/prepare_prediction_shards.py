#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare contiguous eval-json shards after an existing JSONL prefix.")
    parser.add_argument("--eval_json", type=Path, required=True)
    parser.add_argument("--existing_jsonl", type=Path, required=True)
    parser.add_argument("--work_dir", type=Path, required=True)
    parser.add_argument("--num_shards", type=int, required=True)
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Optional comma/space separated positive shard weights, one per requested shard.",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def copy_valid_prefix(src: Path, dst: Path) -> int:
    count = 0
    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w", encoding="utf-8") as out:
        if not src.exists():
            return 0
        with src.open("r", encoding="utf-8") as handle:
            for line in handle:
                text = line.strip()
                if not text:
                    continue
                try:
                    json.loads(text)
                except json.JSONDecodeError:
                    break
                out.write(text + "\n")
                count += 1
    return count


def parse_weights(raw: str | None, num_shards: int) -> list[float] | None:
    if raw is None or not raw.strip():
        return None
    parts = raw.replace(",", " ").split()
    if len(parts) != num_shards:
        raise ValueError(f"--weights length must match --num_shards: {len(parts)} != {num_shards}")
    weights = [float(part) for part in parts]
    if any(weight <= 0 for weight in weights):
        raise ValueError("--weights must all be positive")
    return weights


def shard_sizes(total: int, num_shards: int, weights: list[float] | None) -> list[int]:
    if weights is None:
        base = total // num_shards
        extra = total % num_shards
        return [base + (1 if idx < extra else 0) for idx in range(num_shards)]

    weight_sum = sum(weights)
    exact = [total * weight / weight_sum for weight in weights]
    sizes = [int(value) for value in exact]
    remaining = total - sum(sizes)
    order = sorted(range(num_shards), key=lambda idx: exact[idx] - sizes[idx], reverse=True)
    for idx in order[:remaining]:
        sizes[idx] += 1
    return sizes


def main() -> None:
    args = parse_args()
    if args.num_shards < 1:
        raise ValueError("--num_shards must be >= 1")
    weights = parse_weights(args.weights, args.num_shards)

    args.work_dir.mkdir(parents=True, exist_ok=True)
    prefix_jsonl = args.work_dir / "prefix.jsonl"
    prefix_count = copy_valid_prefix(args.existing_jsonl, prefix_jsonl)

    examples = json.loads(args.eval_json.read_text(encoding="utf-8"))
    total = len(examples)
    prefix_count = min(prefix_count, total)
    remaining = total - prefix_count

    shards = []
    chunk_dir = args.work_dir / "chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    if remaining > 0:
        start = prefix_count
        for idx, size in enumerate(shard_sizes(remaining, args.num_shards, weights)):
            if size <= 0:
                continue
            end = start + size
            eval_path = chunk_dir / f"chunk_{idx:03d}_{start}_{end}.json"
            pred_path = chunk_dir / f"chunk_{idx:03d}_{start}_{end}.jsonl"
            if args.force or not eval_path.exists():
                eval_path.write_text(
                    json.dumps(examples[start:end], ensure_ascii=False),
                    encoding="utf-8",
                )
            shards.append(
                {
                    "index": idx,
                    "start": start,
                    "end": end,
                    "expected_rows": size,
                    "eval_json": str(eval_path.resolve()),
                    "pred_jsonl": str(pred_path.resolve()),
                }
            )
            start = end

    manifest = {
        "eval_json": str(args.eval_json.resolve()),
        "existing_jsonl": str(args.existing_jsonl.resolve()),
        "prefix_jsonl": str(prefix_jsonl.resolve()),
        "prefix_count": int(prefix_count),
        "total": int(total),
        "remaining": int(remaining),
        "num_shards_requested": int(args.num_shards),
        "weights": weights,
        "shards": shards,
    }
    manifest_path = args.work_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
