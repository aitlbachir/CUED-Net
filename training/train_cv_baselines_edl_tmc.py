#!/usr/bin/env python
"""
train_cv_baselines_edl_tmc.py — 5x5 CV for the evidential baselines
(Single-view EDL, TMC), using the IDENTICAL fold loader, augmentation,
schedule, and CSV schema as train_cv.py / train_cv_baselines.py.

Single-view EDL trains ONE encoder on the CC view (img_mlo ignored).
TMC trains two evidential encoders fused by Dempster's rule at inference.

Both use evidential losses matching CUED-Net's per-view term exactly.

LOCKED CSV SCHEMA:
    model, seed, fold, patient_id, label, prob_malignant, predicted, uncertainty

USAGE
  Single-view EDL (25 runs):
    python train_cv_baselines_edl_tmc.py --method single_view_edl \
        --seeds 42,123,456,789,2024 --epochs 50 --patience 15 \
        --output_dir /workspace/cued_net/cv_edl \
        --pred_csv  /workspace/cued_net/cv_preds/single_view_edl_preds.csv

  TMC (25 runs):
    python train_cv_baselines_edl_tmc.py --method tmc \
        --seeds 42,123,456,789,2024 --epochs 50 --patience 15 \
        --output_dir /workspace/cued_net/cv_tmc \
        --pred_csv  /workspace/cued_net/cv_preds/tmc_preds.csv

  Smoke ONE fold:  add  --seeds 42 --only_fold 0
"""

import argparse, csv, json, sys, time
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
from baseline_models_edl_tmc import (SingleViewEDL, SingleViewEDLLoss,
                                     TMCDualView, TMCLoss,
                                     edl_predict, tmc_predict)


def build_optimizer(model, lr, single_view=False):
    groups = [
        {"params": model.encoder_cc.features.parameters(), "lr": lr * 0.1},
        {"params": model.encoder_cc.classifier.parameters(), "lr": lr},
        {"params": model.encoder_cc.evidential.parameters(), "lr": lr},
    ]
    if not single_view:
        groups += [
            {"params": model.encoder_mlo.features.parameters(), "lr": lr * 0.1},
            {"params": model.encoder_mlo.classifier.parameters(), "lr": lr},
            {"params": model.encoder_mlo.evidential.parameters(), "lr": lr},
        ]
    return optim.AdamW(groups, weight_decay=1e-4)


def train_one_model(loaders, class_weights, device, args, tag=""):
    single = (args.method == "single_view_edl")
    if single:
        model = SingleViewEDL(num_classes=2, pretrained=True).to(device)
        criterion = SingleViewEDLLoss(num_classes=2)
    else:
        model = TMCDualView(num_classes=2, pretrained=True).to(device)
        criterion = TMCLoss(num_classes=2)

    optimizer = build_optimizer(model, args.lr, single_view=single)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)
    scaler = GradScaler("cuda")
    cw = class_weights.to(device)

    best_metric, best_state, patience = 0.0, None, 0
    for epoch in range(1, args.epochs + 1):
        model.freeze_encoders() if epoch <= 5 else model.unfreeze_encoders()

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
            scaler.step(optimizer); scaler.update()
        scheduler.step()

        # val (deterministic) for early stopping
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
            best_metric, patience = combined, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= args.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def evaluate_predictions(ys, ps):
    preds = (np.asarray(ps) >= 0.5).astype(int)
    return {"f1": float(f1_score(ys, preds, zero_division=0)),
            "auc": float(roc_auc_score(ys, ps)) if len(np.unique(ys)) > 1 else 0.5,
            "accuracy": float(accuracy_score(ys, preds))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=["single_view_edl", "tmc"])
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
    model_name = "Single-view-EDL" if args.method == "single_view_edl" else "TMC"
    predict = edl_predict if args.method == "single_view_edl" else tmc_predict

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
            model = train_one_model(loaders, cw, device, args, tag=args.method[:4])

            ys, ps, us, pids = [], [], [], []
            with torch.no_grad():
                for batch in loaders["val"]:
                    cc = batch["img_cc"].to(device); mlo = batch["img_mlo"].to(device)
                    out = predict(model, cc, mlo)
                    ps.append(out["prob"][:, 1].cpu().numpy())
                    us.append(out["uncertainty"].cpu().numpy())
                    ys.append(batch["label"].numpy())
                    bp = batch.get("patient_id", ["?"] * len(batch["label"]))
                    pids.extend(list(bp))
            ys = np.concatenate(ys); ps = np.concatenate(ps); us = np.concatenate(us)

            for i in range(len(ys)):
                writer.writerow({"model": model_name, "seed": seed, "fold": fold,
                                 "patient_id": pids[i], "label": int(ys[i]),
                                 "prob_malignant": float(ps[i]),
                                 "predicted": int(ps[i] >= 0.5),
                                 "uncertainty": float(us[i])})
            fh.flush()

            m = evaluate_predictions(ys, ps)
            summary.append((seed, fold, m["f1"], m["auc"], m["accuracy"]))
            print(f"[cv] {model_name} seed={seed} fold={fold} -> "
                  f"F1={m['f1']:.4f} AUC={m['auc']:.4f}")
            del model; torch.cuda.empty_cache()

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
              f"F1={out['cv_summary']['f1_mean']:.4f}+/-{out['cv_summary']['f1_std']:.4f}  "
              f"AUC={out['cv_summary']['auc_mean']:.4f}+/-{out['cv_summary']['auc_std']:.4f}")
    out["minutes"] = (time.time() - t0) / 60
    json.dump(out, open(Path(args.output_dir) / "cv_results.json", "w"), indent=2)
    print(f"[cv] preds -> {args.pred_csv}")
    print(f"[cv] summary -> {Path(args.output_dir)/'cv_results.json'}  ({out['minutes']:.1f} min)")


if __name__ == "__main__":
    main()
