#!/usr/bin/env python
"""Train one (method, outer fold, seed) under Protocol P2.

Early stopping uses the inner validation split only. The outer fold is scored
once, after the best inner-validation checkpoint is restored. Fold -1 is the
test phase: training uses the whole development cohort and no test case is read.
"""

import argparse
import csv
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score, roc_auc_score
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.p2_data import ImageCache, build_splits, check_splits, load_manifest, make_loaders
from models.baseline_models import (CrossViewTransformer, SingleViewEDL, SingleViewEDLLoss, SoftmaxDualLoss,
                                    SoftmaxDualView, TMCDualView, TMCLoss, mc_dropout_predict)
from models.cued_net import CUEDNet, CUEDNetLoss
from training.p2_config import LOSS, MC_DROPOUT_PASSES, METHODS, TRAINING


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build(method, pretrained=True):
    cfg = METHODS[method]
    kind = cfg["model"]
    if kind == "cued_net":
        model = CUEDNet(pretrained=pretrained, backbone=cfg["backbone"])
        crit = CUEDNetLoss(lambda_kl=LOSS["lambda_kl"], annealing_epochs=LOSS["annealing_epochs"],
                           lambda_consistency=cfg.get("lambda_consistency", LOSS["lambda_consistency"]),
                           lambda_vdl=cfg.get("lambda_vdl", 0.0), kl_mode=cfg["kl_mode"])
    elif kind == "softmax_dual":
        model, crit = SoftmaxDualView(pretrained=pretrained), SoftmaxDualLoss()
    elif kind == "cross_view":
        model, crit = CrossViewTransformer(pretrained=pretrained), SoftmaxDualLoss()
    elif kind == "tmc":
        model, crit = TMCDualView(pretrained=pretrained), TMCLoss(LOSS["lambda_kl"], LOSS["annealing_epochs"])
    elif kind == "sv_edl":
        model, crit = SingleViewEDL(pretrained=pretrained), SingleViewEDLLoss(LOSS["lambda_kl"], LOSS["annealing_epochs"])
    else:
        raise ValueError(kind)
    return model, crit


def param_groups(model):
    t = TRAINING
    encoder, heads = [], []
    for name, p in model.named_parameters():
        (encoder if ".features." in name else heads).append(p)
    return [{"params": encoder, "lr": t["lr_encoder"]}, {"params": heads, "lr": t["lr_head"]}]


def per_case(model, cc, mlo):
    """Everything saved for one batch, as a dict of 1-D numpy arrays."""
    out = model(cc, mlo)
    row = {"p": out["prob"][:, 1]}
    if "cc_out" in out:
        c, m = out["cc_out"], out["mlo_out"]
        row.update(p_cc=c["prob"][:, 1], p_mlo=m["prob"][:, 1], s_cc=c["strength"], s_mlo=m["strength"],
                   u_cc=c["uncertainty"], u_mlo=m["uncertainty"], d_sym=out["d_sym"],
                   w_cc=out["fusion_weights"]["cc"], w_mlo=out["fusion_weights"]["mlo"])
    if "uncertainty" in out:
        row["vacuity"] = out["uncertainty"]
    if "prob_cc" in out:
        row.update(p_cc=out["prob_cc"][:, 1], p_mlo=out["prob_mlo"][:, 1])
    return {k: v.detach().float().cpu().numpy() for k, v in row.items()}


@torch.no_grad()
def predict(model, loader, device, mc_passes=0):
    model.eval()
    collected, labels, indices = {}, [], []
    for batch in loader:
        cc, mlo = batch["img_cc"].to(device), batch["img_mlo"].to(device)
        with autocast(device.type, enabled=device.type == "cuda"):
            row = per_case(model, cc, mlo)
            if mc_passes:
                mc_p, mc_h = mc_dropout_predict(model, cc, mlo, T=mc_passes)
                row["mc_p"] = mc_p.float().cpu().numpy()
                row["mc_entropy"] = mc_h.float().cpu().numpy()
        for k, v in row.items():
            collected.setdefault(k, []).append(v)
        labels.append(np.asarray(batch["label"]))
        indices.append(np.asarray(batch["index"]))
    res = {k: np.concatenate(v) for k, v in collected.items()}
    res["label"], res["index"] = np.concatenate(labels), np.concatenate(indices)
    return res


def selection_score(pred):
    y, p = pred["label"], pred["p"]
    auc = roc_auc_score(y, p) if len(set(y.tolist())) > 1 else 0.5
    return TRAINING["selection_f1_weight"] * f1_score(y, p >= 0.5, zero_division=0) + TRAINING["selection_auc_weight"] * auc


def write_csv(path, pred, pairs, method, fold, seed, split):
    keys = [k for k in pred if k not in ("label", "index")]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "fold", "seed", "split", "pair_key", "patient_id", "label"] + keys)
        for i, idx in enumerate(pred["index"]):
            p = pairs[int(idx)]
            w.writerow([method, fold, seed, split, p["key"], p["patient_id"], int(pred["label"][i])]
                       + [f"{float(pred[k][i]):.7g}" for k in keys])


def train(args):
    t0 = time.time()
    out = Path(args.out) / args.method / f"fold_{args.fold}" / f"seed_{args.seed}"
    if (out / "done.json").exists():
        print("already done", out)
        return
    out.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest(args.manifest)
    splits = build_splits(manifest)
    check_splits(manifest, splits)
    split = splits[args.fold]
    cache = ImageCache(args.patches)
    cache.preload(manifest["pairs"])

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, criterion = build(args.method, pretrained=not args.no_pretrained)
    model = model.to(device)
    loaders = make_loaders(manifest, split, cache, args.seed, TRAINING["batch_size"], args.workers)

    t = TRAINING
    opt = torch.optim.AdamW(param_groups(model), weight_decay=t["weight_decay"])
    sched = CosineAnnealingWarmRestarts(opt, T_0=t["cosine_t0"], T_mult=t["cosine_tmult"], eta_min=t["cosine_eta_min"])
    scaler = GradScaler(device.type, enabled=device.type == "cuda")

    best, best_state, stale, log = -1.0, None, 0, []
    for epoch in range(1, args.epochs + 1):
        ep0 = time.time()
        (model.freeze_encoders if epoch <= t["freeze_epochs"] else model.unfreeze_encoders)()
        model.train()
        for i, batch in enumerate(loaders["train"]):
            if args.max_batches and i >= args.max_batches:
                break
            cc, mlo = batch["img_cc"].to(device, non_blocking=True), batch["img_mlo"].to(device, non_blocking=True)
            y = batch["label"].to(device)
            opt.zero_grad(set_to_none=True)
            with autocast(device.type, enabled=device.type == "cuda"):
                outputs = model(cc, mlo)
            # Digamma and lgamma terms are computed in float32.
            outputs = to_float(outputs)
            with autocast(device.type, enabled=False):
                loss = criterion(outputs, y, epoch)["total"]
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), t["grad_clip"])
            scaler.step(opt)
            scaler.update()
        sched.step()
        val = predict(model, loaders["inner_val"], device)
        score = selection_score(val)
        log.append(dict(epoch=epoch, score=float(score), loss=float(loss), seconds=time.time() - ep0))
        if score > best:
            best, stale = score, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= t["patience"]:
                break

    model.load_state_dict(best_state)
    mc = MC_DROPOUT_PASSES if METHODS[args.method].get("mc_dropout") else 0
    pairs = manifest["pairs"]
    write_csv(out / "inner_val.csv", predict(model, loaders["inner_val"], device, mc), pairs, args.method, args.fold, args.seed, "inner_val")
    if "outer_val" in loaders:
        write_csv(out / "outer_val.csv", predict(model, loaders["outer_val"], device, mc), pairs, args.method, args.fold, args.seed, "outer_val")
    if METHODS[args.method].get("save_ckpt") and (args.fold == -1 or args.method == "cued_net"):
        torch.save({"model_state_dict": best_state, "method": args.method, "fold": args.fold, "seed": args.seed},
                   out / "best_model.pt")
    json.dump(dict(method=args.method, fold=args.fold, seed=args.seed, best_score=best,
                   best_epoch=int(np.argmax([e["score"] for e in log])) + 1, epochs_run=len(log),
                   seconds=time.time() - t0, device=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
                   manifest_md5=manifest.get("md5", ""), log=log), open(out / "done.json", "w"))
    print(f"done {args.method} fold={args.fold} seed={args.seed} score={best:.4f} "
          f"epochs={len(log)} {time.time() - t0:.0f}s", flush=True)


def to_float(x):
    if isinstance(x, dict):
        return {k: to_float(v) for k, v in x.items()}
    if torch.is_tensor(x) and x.is_floating_point():
        return x.float()
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=sorted(METHODS))
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--patches", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=TRAINING["max_epochs"])
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--max_batches", type=int, default=0, help="smoke tests only")
    ap.add_argument("--no_pretrained", action="store_true", help="smoke tests only")
    train(ap.parse_args())


if __name__ == "__main__":
    main()
