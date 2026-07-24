import copy
import json
import os
import time

import torch

import Procedure
import register
import utils
import world
from register import dataset
from world import cprint


def metric_value(results, name="overall_ndcg@20"):
    if name in {"strict_warmup_ndcg@20", "cold_warmup_ndcg@20"}:
        return 0.5 * float(results[0]["ndcg"][0]) + 0.5 * float(results[1]["ndcg"][0])
    if "," in name:
        parts = [part.strip() for part in name.split(",") if part.strip()]
        if not parts:
            raise ValueError(f"Unsupported best metric name: {name}")
        return sum(metric_value(results, part) for part in parts) / len(parts)
    split_name = None
    metric_at_k = None
    for prefix in ("strict_cold", "warmup", "overall", "strict", "cold", "warm"):
        marker = prefix + "_"
        if name.startswith(marker):
            split_name = prefix
            metric_at_k = name[len(marker) :]
            break
    if split_name is None or metric_at_k is None:
        raise ValueError(f"Unsupported best metric name: {name}")
    metric, _at = metric_at_k.split("@", 1)
    split_to_idx = {
        "strict": 0,
        "strict_cold": 0,
        "cold": 0,
        "warmup": 1,
        "warm": 2,
        "overall": 3,
    }
    idx = split_to_idx[split_name]
    return float(results[idx][metric][0])


def metric_objective(results, best):
    thresholds = best.get("thresholds") or {}
    if thresholds:
        values = {name: metric_value(results, name) for name in thresholds}
        margins = {name: values[name] - threshold for name, threshold in thresholds.items()}
        min_margin = min(margins.values())
        tie_break = sum(values.values()) * 1e-9
        best["metric_values"] = values
        best["metric_margins"] = margins
        return min_margin + tie_break, min_margin
    value = metric_value(results, best["metric"])
    target = best.get("target")
    if target is None:
        return value, value
    objective = -abs(value - target) / max(abs(target), 1e-12)
    return objective, value


def parse_thresholds(text):
    thresholds = {}
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"Invalid FINALUPDATE_BEST_THRESHOLDS entry: {part!r}")
        name, value = part.split("=", 1)
        thresholds[name.strip()] = float(value)
    return thresholds


def evaluate_and_maybe_save(epoch, mode, recmodel, weight_file, best):
    cprint(f"[{mode.upper()}]")
    recmodel.eval()
    with torch.no_grad():
        results = Procedure.Test(dataset, recmodel, mode=mode)
    current_objective, current_value = metric_objective(results, best)
    target_text = "" if best.get("target") is None else f" target={best['target']:.8f}"
    print(
        f"[BEST-CHECK] epoch={epoch} mode={mode} {best['metric']}={current_value:.8f}"
        f"{target_text} objective={current_objective:.8f}"
    )
    if mode == "val" and current_objective > best["score"]:
        best["score"] = current_objective
        best["metric_value"] = current_value
        best["epoch"] = epoch
        best["state_dict"] = copy.deepcopy({k: v.detach().cpu() for k, v in recmodel.state_dict().items()})
        torch.save(best["state_dict"], weight_file)
        print(
            f"[BEST-UPDATE] epoch={epoch} {best['metric']}={current_value:.8f}"
            f"{target_text} objective={current_objective:.8f} saved={weight_file}"
        )
    return results


def main():
    utils.set_seed(world.seed)
    print(">>SEED:", world.seed)

    recmodel = register.MODELS[world.model_name](world.config, dataset)
    recmodel = recmodel.to(world.device)
    bpr = utils.BPRLoss(recmodel, world.config, dataset)

    weight_file = utils.getFileName()
    print(f"load and save best to {weight_file}")
    print(f"user number:{recmodel.num_users}, item number:{recmodel.num_items} ")
    if world.LOAD:
        try:
            recmodel.load_state_dict(torch.load(weight_file, map_location=torch.device("cpu")))
        except FileNotFoundError:
            print(f"{weight_file} not exists, start from beginning")

    valid_every = max(int(os.environ.get("FINALUPDATE_VALID_EVERY", "1")), 1)
    best_metric = os.environ.get("FINALUPDATE_BEST_METRIC", "overall_ndcg@20")
    best_target_env = os.environ.get("FINALUPDATE_BEST_TARGET", "").strip()
    best_target = float(best_target_env) if best_target_env else None
    best_thresholds = parse_thresholds(os.environ.get("FINALUPDATE_BEST_THRESHOLDS", "").strip())
    best = {
        "metric": best_metric,
        "target": best_target,
        "thresholds": best_thresholds,
        "score": float("-inf"),
        "metric_value": None,
        "metric_values": None,
        "metric_margins": None,
        "epoch": 0,
        "state_dict": None,
    }

    print(
        f"[BEST-CONFIG] valid_every={valid_every} best_metric={best_metric} "
        f"best_target={best_target} best_thresholds={best_thresholds}"
    )
    cprint("not enable tensorflowboard")

    evaluate_and_maybe_save(0, "val", recmodel, weight_file, best)

    for epoch in range(world.TRAIN_epochs):
        start = time.time()
        output_information = Procedure.BPR_train_original(dataset, recmodel, bpr, epoch, neg_k=1, w=None)
        print(f"EPOCH[{epoch + 1}/{world.TRAIN_epochs}] {output_information}")
        if (epoch + 1) % valid_every == 0 or (epoch + 1) == world.TRAIN_epochs:
            evaluate_and_maybe_save(epoch + 1, "val", recmodel, weight_file, best)
        print(f"[EPOCH-TIME] epoch={epoch + 1} seconds={time.time() - start:.2f}")

    if best["state_dict"] is None:
        torch.save(recmodel.state_dict(), weight_file)
        best["epoch"] = world.TRAIN_epochs
        best["score"] = float("nan")
    else:
        recmodel.load_state_dict({k: v.to(world.device) for k, v in best["state_dict"].items()})

    meta = {
        "seed": world.seed,
        "dataset": world.dataset,
        "model": world.model_name,
        "epochs": world.TRAIN_epochs,
        "valid_every": valid_every,
        "best_metric": best["metric"],
        "best_target": best["target"],
        "best_thresholds": best["thresholds"],
        "best_epoch": best["epoch"],
        "best_score": best["score"],
        "best_metric_value": best["metric_value"],
        "best_metric_values": best["metric_values"],
        "best_metric_margins": best["metric_margins"],
        "weight_file": weight_file,
    }
    meta_path = weight_file + ".best_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"[BEST-META] {json.dumps(meta, sort_keys=True)}")

    cprint("[VAL]")
    recmodel.eval()
    with torch.no_grad():
        Procedure.Test(dataset, recmodel, mode="val")

    cprint("[TEST]")
    recmodel.eval()
    with torch.no_grad():
        Procedure.Test(dataset, recmodel, mode="test")


if __name__ == "__main__":
    main()
