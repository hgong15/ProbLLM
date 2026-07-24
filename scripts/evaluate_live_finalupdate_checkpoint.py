#!/usr/bin/env python3
import argparse
import json
import os
import re
import shlex
import sys
from pathlib import Path


def parse_main_command(seed_dir: Path):
    commands_path = seed_dir / "commands.txt"
    if not commands_path.exists():
        raise FileNotFoundError(f"commands.txt not found: {commands_path}")

    main_lines = [
        line.strip()
        for line in commands_path.read_text(encoding="utf-8", errors="replace").splitlines()
        if "main_best.py" in line and " --dataset " in line
    ]
    if not main_lines:
        raise ValueError(f"No main_best.py command found in {commands_path}")

    line = main_lines[-1]
    match = re.search(r"\] cd (.+?) && (.+)$", line)
    if not match:
        raise ValueError(f"Could not parse command line: {line}")

    workdir = Path(match.group(1)).resolve()
    tokens = shlex.split(match.group(2))
    try:
        main_idx = next(i for i, token in enumerate(tokens) if token.endswith("main_best.py"))
    except StopIteration as exc:
        raise ValueError(f"main_best.py token not found in: {line}") from exc
    main_args = tokens[main_idx + 1 :]
    return workdir, main_args


def get_arg(tokens, name, default=None):
    try:
        idx = tokens.index(name)
    except ValueError:
        return default
    if idx + 1 >= len(tokens):
        return default
    return tokens[idx + 1]


def metrics_to_json(results):
    split_names = ["strict_cold", "warmup", "warm", "overall"]
    payload = {}
    for split, result in zip(split_names, results):
        payload[split] = {}
        for metric, values in result.items():
            if len(values) > 0:
                payload[split][f"{metric}@20"] = float(values[0])
    return payload


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a live FinalUpdate best checkpoint without touching the training process."
    )
    parser.add_argument("--seed_dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--mode", choices=["val", "test"], default="test")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()

    seed_dir = args.seed_dir.resolve()
    workdir, main_args = parse_main_command(seed_dir)
    runroot = workdir.parent
    model_name = get_arg(main_args, "--model", "lgn")
    file_name = get_arg(main_args, "--file_name")
    if not file_name:
        raise ValueError("Could not parse --file_name from main_best.py command")

    checkpoint = args.checkpoint
    if checkpoint is None:
        checkpoint = runroot / "code" / "checkpoints" / f"{model_name}-{file_name}.pth.tar"
    checkpoint = checkpoint.resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")

    if args.device == "cpu":
        os.environ["FINALUPDATE_FORCE_CPU"] = "1"
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

    finalupdate_dir = workdir.resolve()
    os.chdir(finalupdate_dir)
    sys.path.insert(0, str(finalupdate_dir))
    sys.argv = ["main_best.py"] + main_args

    import torch

    import Procedure
    import register
    import utils
    import world

    utils.set_seed(world.seed)
    recmodel = register.MODELS[world.model_name](world.config, register.dataset)
    recmodel = recmodel.to(world.device)
    state = torch.load(checkpoint, map_location=torch.device("cpu"))
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    recmodel.load_state_dict(state)
    recmodel.eval()

    with torch.no_grad():
        results = Procedure.Test(register.dataset, recmodel, mode=args.mode)

    payload = {
        "seed_dir": str(seed_dir),
        "runroot": str(runroot),
        "checkpoint": str(checkpoint),
        "mode": args.mode,
        "device": str(world.device),
        "metrics": metrics_to_json(results),
    }
    meta_path = Path(str(checkpoint) + ".best_meta.json")
    try:
        if meta_path.exists():
            payload["best_meta"] = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        payload["best_meta_error"] = f"Could not parse {meta_path}"

    output = args.output
    if output is None:
        output = seed_dir / f"live_{args.mode}_metrics.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    m = payload["metrics"]
    print(
        "LIVE-EVAL "
        f"{seed_dir.parent.name}/{seed_dir.name} "
        f"overall={m.get('overall', {}).get('recall@20'):.6f}/"
        f"{m.get('overall', {}).get('ndcg@20'):.6f} "
        f"strict={m.get('strict_cold', {}).get('recall@20'):.6f}/"
        f"{m.get('strict_cold', {}).get('ndcg@20'):.6f} "
        f"warmup={m.get('warmup', {}).get('recall@20'):.6f}/"
        f"{m.get('warmup', {}).get('ndcg@20'):.6f}"
    )


if __name__ == "__main__":
    main()
