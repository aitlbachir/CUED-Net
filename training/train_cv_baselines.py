#!/usr/bin/env python
"""Cross-validation training for MC-Dropout and Deep-Ensemble baselines."""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.amp import GradScaler, autocast
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from tqdm import tqdm

sys.path.insert(0, "/workspace/cued_net")
from train_cued_net import set_seed
from cv_dataloaders import get_cv_dataloaders
from baseline_models import (SoftmaxDualView, SoftmaxDualLoss,
                             mc_dropout_predict, ensemble_predict)


def build_optimizer(model, lr):
    return optim.AdamW([
        {"params": model.encoder_cc.features.parameters(), "lr": lr * 0.1},
        {"params": model.encoder_mlo.features.parameters(), "lr": lr * 0.1},
        {"params": model.encoder_cc.classifier.parameters(), "lr": lr},
        {"params": model.encoder_mlo.classifier.parameters(), "lr": lr},
        {"params": model.encoder_cc.head.parameters(), "lr": lr},
        {"params": model.encoder_mlo.head.parameters(), "lr": lr},
    ], weight_decay=1e-4)


def train_one_model(loaders, class_weights, device, args, tag=""):
    """Train a single SoftmaxDualView on the fold; return best-val model."""
    model = SoftmaxDualView(num_classes=2, pretrained=True).to(device)
    criterion = SoftmaxDualLoss(num_classes=2)
    optimizer = build_optimizer(model, args.lr)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)
    scaler = GradScaler("cuda")
    cw = class_weights.to(device)

    best_metric, best_state, patience = 0.0, None, 0
    for epoch in range(1, args.epochs + 1):
        if epoch <= 5:
            model.freeze_encoders()
        else:
            model.unfreeze_encoders()

        model.train()
        for batch in tqdm(loaders["train"], desc=f"{tag} ep{epoch}", leave=False):
            cc = batch["img_cc"].to(device)
            mlo = batch["img_mlo"].to(device)
            y = batch["label"].to(device)
            optimizer.zero_grad()
            with autocast("cuda"):
                out = model(cc, mlo)
                loss = criterion(out, y, epoch, cw)["total"]
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        scheduler.step()

        # val (deterministic forward) for early stopping
        model.eval()
        ys, ps = [], []
        with torch.no_grad():
            for batch in loaders["val"]:
                cc = batch["img_cc"].to(device); mlo = batch["img_mlo"].to(device)
                out = model(cc, mlo)
                ps.append(out["prob"][:, 1].cpu().numpy()); ys.append(batch["label"].numpy())
        ys = np.concatenate(ys); ps = np.concatenate(ps)
        preds = (ps >= 0.5).astype(int)
        f1 = f1_score(ys, preds, zero_division=0)
        auc = roc_auc_score(ys, ps) if len(np.unique(ys)) > 1 else 0.5
        combined = 0.6 * f1 + 0.4 * auc
        if combined > best_metric:
            best_metric = combined
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= args.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def evaluate_predictions(ys, ps):
    preds = (np.asarray(ps) >= 0.5).astype(int)
    return {
        "f1": float(f1_score(ys, preds, zero_division=0)),
        "auc": float(roc_auc_score(ys, ps)) if len(np.unique(ys)) > 1 else 0.5,
        "accuracy": float(accuracy_score(ys, preds)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=["mc_dropout", "deep_ensemble"])
    ap.add_argument("--ensemble_M", type=int, default=5)
    ap.add_argument("--mc_T", type=int, default=50)
    ap.add_argument("--data_root", default="/workspace/cbis-ddsm")
    ap.add_argument("--folds_json", default="/workspace/cued_net/cv_folds.json")
    ap.add_argument("--seeds", default="42,123,456,789,2024")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--pred_csv", required=True)
    ap.add_argument("--only_fold", type=int, default=-1)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seeds = [int(s) for s in args.seeds.split(",")]
    n_folds = len(json.load(open(args.folds_json))["folds"])
    folds = range(n_folds) if args.only_fold < 0 else [args.only_fold]
    model_name = "MC-Dropout" if args.method == "mc_dropout" else f"Deep-Ensemble(M={args.ensemble_M})"

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.pred_csv).parent.mkdir(parents=True, exist_ok=True)
    fields = ["model", "seed", "fold", "patient_id", "label",
              "prob_malignant", "predicted", "uncertainty"]
    write_header = not Path(args.pred_csv).exists()
    fh = open(args.pred_csv, "a", newline="")
    writer = csv.DictWriter(fh, fieldnames=fields)
    if write_header:
        writer.writeheader()

    summary, t0 = [], time.time()
    for seed in seeds:
        for fold in folds:
            print(f"\n{'#'*60}\n# {model_name}  seed={seed} fold={fold}\n{'#'*60}")
            set_seed(seed)
            loaders, cw = get_cv_dataloaders(args.data_root, args.folds_json, fold,
                                             batch_size=args.batch_size, oversample=True)

            # --- train ---
            if args.method == "mc_dropout":
                models = [train_one_model(loaders, cw, device, args, tag="mcd")]
            else:
                models = []
                for mi in range(args.ensemble_M):
                    set_seed(seed * 100 + mi)   # distinct init per ensemble member
                    models.append(train_one_model(loaders, cw, device, args, tag=f"ens{mi}"))

            # --- inference w/ UQ on the fold val set ---
            ys, ps, us, pids = [], [], [], []
            with torch.no_grad():
                for batch in loaders["val"]:
                    cc = batch["img_cc"].to(device); mlo = batch["img_mlo"].to(device)
                    if args.method == "mc_dropout":
                        out = mc_dropout_predict(models[0], cc, mlo, T=args.mc_T)
                    else:
                        out = ensemble_predict(models, cc, mlo)
                    ps.append(out["prob"][:, 1].cpu().numpy())
                    us.append(out["uncertainty"].cpu().numpy())
                    ys.append(batch["label"].numpy())
                    bp = batch.get("patient_id", ["?"] * len(batch["label"]))
                    pids.extend(list(bp))
            ys = np.concatenate(ys); ps = np.concatenate(ps); us = np.concatenate(us)

            for i in range(len(ys)):
                writer.writerow({
                    "model": model_name, "seed": seed, "fold": fold,
                    "patient_id": pids[i], "label": int(ys[i]),
                    "prob_malignant": float(ps[i]),
                    "predicted": int(ps[i] >= 0.5),
                    "uncertainty": float(us[i]),
                })
            fh.flush()

            m = evaluate_predictions(ys, ps)
            summary.append((seed, fold, m["f1"], m["auc"], m["accuracy"]))
            print(f"[cv] {model_name} seed={seed} fold={fold} -> F1={m['f1']:.4f} AUC={m['auc']:.4f}")

            del models
            torch.cuda.empty_cache()

    fh.close()

    arr = np.array([(s, f, f1, au) for (s, f, f1, au, _) in summary], dtype=float)
    per_seed_f1, per_seed_auc = [], []
    for seed in seeds:
        mm = arr[arr[:, 0] == seed]
        if len(mm):
            per_seed_f1.append(mm[:, 2].mean()); per_seed_auc.append(mm[:, 3].mean())
    out = {"model": model_name,
           "per_run": [{"seed": int(s), "fold": int(f), "f1": f1, "auc": au, "acc": ac}
                       for (s, f, f1, au, ac) in summary]}
    if per_seed_f1:
        out["cv_summary"] = {
            "f1_mean": float(np.mean(per_seed_f1)),
            "f1_std": float(np.std(per_seed_f1, ddof=1)) if len(per_seed_f1) > 1 else 0.0,
            "auc_mean": float(np.mean(per_seed_auc)),
            "auc_std": float(np.std(per_seed_auc, ddof=1)) if len(per_seed_auc) > 1 else 0.0,
            "n_seeds": len(per_seed_f1), "n_folds": len(folds)}
        print(f"\n[cv] {model_name} 5x5: "
              f"F1={out['cv_summary']['f1_mean']:.4f}±{out['cv_summary']['f1_std']:.4f}  "
              f"AUC={out['cv_summary']['auc_mean']:.4f}±{out['cv_summary']['auc_std']:.4f}")
    out["minutes"] = (time.time() - t0) / 60
    json.dump(out, open(Path(args.output_dir) / "cv_results.json", "w"), indent=2)
    print(f"[cv] preds -> {args.pred_csv}\n[cv] summary -> {Path(args.output_dir)/'cv_results.json'}"
          f"  ({out['minutes']:.1f} min)")


if __name__ == "__main__":
    main()
