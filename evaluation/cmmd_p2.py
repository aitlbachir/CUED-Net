#!/usr/bin/env python
"""External evaluation on CMMD, zero-shot and after head adaptation.

Zero-shot: the fold -1 models, unchanged, on every pair. Temperature and all
thresholds come from CBIS-DDSM inner validation.
Adapted: encoders frozen, heads retrained on adaptation patients, selection on
adaptation validation patients, reported on evaluation patients only.
"""

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.p2_data import ImageCache, PairDataset, eval_transform, load_manifest, make_loaders
from training.p2_config import ADAPTATION, ADAPTED_METHODS, LOCKED_METHODS, LOSS, MC_DROPOUT_PASSES, METHODS, SEEDS
from training.train_p2 import build, predict, selection_score, set_seed, to_float, write_csv


def subset(pairs, name):
    idx = {"cmmd": range(len(pairs)),
           "cmmd_eval": [i for i, p in enumerate(pairs) if p["split"] == "eval"],
           "cmmd_mass": [i for i, p in enumerate(pairs) if p["split"] == "eval" and p["contains_mass"]],
           "adapt_train": [i for i, p in enumerate(pairs) if p["split"] == "adapt_train"],
           "adapt_val": [i for i, p in enumerate(pairs) if p["split"] == "adapt_val"]}[name]
    return list(idx)


def loader(pairs, idx, cache):
    return DataLoader(PairDataset(pairs, idx, cache, eval_transform()), batch_size=32, num_workers=2, pin_memory=True)


def load(method, runs, seed, device):
    model, criterion = build(method, pretrained=False)
    ckpt = Path(runs) / method / "fold_-1" / f"seed_{seed}" / "best_model.pt"
    model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=False)["model_state_dict"])
    return model.to(device), criterion


def freeze_for_adaptation(model):
    model.freeze_encoders()
    return [p for p in model.parameters() if p.requires_grad]


def adapt(model, criterion, pairs, cache, seed, device):
    """Retrain the heads. Encoder batch norm stays in evaluation mode."""
    a = ADAPTATION
    set_seed(seed)
    params = freeze_for_adaptation(model)
    opt = torch.optim.AdamW(params, lr=a["lr"], weight_decay=a["weight_decay"])
    split = dict(train=subset(pairs, "adapt_train"), inner_val=subset(pairs, "adapt_val"), outer_val=[])
    loaders = make_loaders(dict(pairs=pairs), split, cache, seed, a["batch_size"], 2)
    scaler = GradScaler(device.type, enabled=device.type == "cuda")
    best, best_state, stale = -1.0, None, 0
    for _ in range(a["max_epochs"]):
        model.train()
        for enc in model._encoders():
            enc.features.eval()
        for batch in loaders["train"]:
            cc, mlo, y = batch["img_cc"].to(device), batch["img_mlo"].to(device), batch["label"].to(device)
            opt.zero_grad(set_to_none=True)
            with autocast(device.type, enabled=device.type == "cuda"):
                out = model(cc, mlo)
            with autocast(device.type, enabled=False):
                loss = criterion(to_float(out), y, LOSS["annealing_epochs"])["total"]
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        score = selection_score(predict(model, loaders["inner_val"], device))
        if score > best:
            best, stale = score, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= a["patience"]:
                break
    model.load_state_dict(best_state)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--patches", required=True)
    ap.add_argument("--runs", required=True, help="folder with fold_-1 checkpoints and inner_val.csv")
    ap.add_argument("--out", required=True)
    ap.add_argument("--phase", choices=["zero_shot", "adapted", "both"], default="both")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pairs = load_manifest(args.manifest)["pairs"]
    groups = [{p["patient_id"] for p in pairs if p["split"] == s} for s in ("adapt_train", "adapt_val", "eval")]
    assert sum(map(len, groups)) == len(set().union(*groups)), "a patient appears in more than one CMMD split"
    cache = ImageCache(args.patches)
    cache.preload(pairs)
    out = Path(args.out)

    if args.phase in ("zero_shot", "both"):
        for method in LOCKED_METHODS:
            mc = MC_DROPOUT_PASSES if METHODS[method].get("mc_dropout") else 0
            for seed in SEEDS:
                model, _ = load(method, args.runs, seed, device)
                pred = predict(model, loader(pairs, subset(pairs, "cmmd"), cache), device, mc)
                d = out / "zero_shot" / method / "fold_-1" / f"seed_{seed}"
                d.mkdir(parents=True, exist_ok=True)
                shutil.copy(Path(args.runs) / method / "fold_-1" / f"seed_{seed}" / "inner_val.csv", d / "inner_val.csv")
                for name in ("cmmd", "cmmd_eval", "cmmd_mass"):
                    keep = np.isin(pred["index"], subset(pairs, name))
                    write_csv(d / f"{name}.csv", {k: v[keep] for k, v in pred.items()}, pairs, method, -1, seed, name)
                print("zero-shot", method, seed, flush=True)

    if args.phase in ("adapted", "both"):
        for method in ADAPTED_METHODS:
            for seed in SEEDS:
                model, criterion = load(method, args.runs, seed, device)
                model = adapt(model, criterion, pairs, cache, seed, device)
                d = out / "adapted" / method / "fold_-1" / f"seed_{seed}"
                d.mkdir(parents=True, exist_ok=True)
                # The adaptation validation patients replace inner validation.
                write_csv(d / "inner_val.csv", predict(model, loader(pairs, subset(pairs, "adapt_val"), cache), device),
                          pairs, method, -1, seed, "inner_val")
                for name in ("cmmd_eval", "cmmd_mass"):
                    write_csv(d / f"{name}.csv", predict(model, loader(pairs, subset(pairs, name), cache), device),
                              pairs, method, -1, seed, name)
                print("adapted", method, seed, flush=True)


if __name__ == "__main__":
    main()
