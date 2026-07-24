#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge JSONL prefix and prediction shards in original order.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output_jsonl", type=Path, required=True)
    return parser.parse_args()


def valid_jsonl_lines(path: Path) -> list[str]:
    lines = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_no}: {exc}") from exc
            lines.append(text)
    return lines


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    output = args.output_jsonl
    output.parent.mkdir(parents=True, exist_ok=True)

    prefix_path = Path(manifest["prefix_jsonl"])
    prefix_lines = valid_jsonl_lines(prefix_path)
    expected_prefix = int(manifest["prefix_count"])
    if len(prefix_lines) != expected_prefix:
        raise ValueError(f"Prefix rows mismatch: expected={expected_prefix} actual={len(prefix_lines)}")

    tmp = output.with_suffix(output.suffix + ".tmp")
    written = 0
    with tmp.open("w", encoding="utf-8") as out:
        for line in prefix_lines:
            out.write(line + "\n")
            written += 1
        for shard in sorted(manifest["shards"], key=lambda item: item["start"]):
            pred_path = Path(shard["pred_jsonl"])
            shard_lines = valid_jsonl_lines(pred_path)
            expected = int(shard["expected_rows"])
            if len(shard_lines) != expected:
                raise ValueError(
                    f"Shard rows mismatch for {pred_path}: expected={expected} actual={len(shard_lines)}"
                )
            for line in shard_lines:
                out.write(line + "\n")
                written += 1

    total = int(manifest["total"])
    if written != total:
        raise ValueError(f"Merged rows mismatch: expected={total} actual={written}")
    os.replace(tmp, output)
    print(json.dumps({"output_jsonl": str(output.resolve()), "rows": written}, indent=2), flush=True)


if __name__ == "__main__":
    main()
