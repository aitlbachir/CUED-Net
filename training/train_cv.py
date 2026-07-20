#!/usr/bin/env python
"""Train CUED-Net under 5-fold x 5-seed cross-validation."""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/workspace/cued_net")
from train_cued_net import train_single_model, evaluate, set_seed
from models.cued_net import CUEDNetLoss
from cv_dataloaders import get_cv_dataloaders


def collect_val_predictions(model, loader, device, criterion, class_weights):
    """Per-sample P(malignant), prediction, uncertainty_total on the val set."""
    model.eval()
    rows = []
    with torch.no_grad():
        for batch in loader:
            cc = batch["img_cc"].to(device)
            mlo = batch["img_mlo"].to(device)
            labels = batch["label"].numpy()
            pids = batch.get("patient_id", ["?"] * len(labels))
            out = model(cc, mlo)
            prob_mal = out["prob"][:, 1].cpu().numpy()
            pred = (prob_mal >= 0.5).astype(int)
            # uncertainty_combined is CUED-Net's fused total; fall back if absent
            u = out.get("uncertainty_combined", out.get("uncertainty_evidential"))
            u = u.cpu().numpy() if u is not None else np.full(len(labels), np.nan)
            for i in range(len(labels)):
                rows.append({
                    "patient_id": pids[i] if isinstance(pids, (list, tuple)) else str(pids[i]),
                    "label": int(labels[i]),
                    "prob_malignant": float(prob_mal[i]),
                    "predicted": int(pred[i]),
                    "uncertainty": float(u[i]),
                })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="/workspace/cbis-ddsm")
    ap.add_argument("--folds_json", default="/workspace/cued_net/cv_folds.json")
    ap.add_argument("--seeds", default="42,123,456,789,2024")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--output_dir", default="/workspace/cued_net/cv_cued_net")
    ap.add_argument("--pred_csv", default="/workspace/cued_net/cv_preds/cued_net_preds.csv")
    ap.add_argument("--model_name", default="CUED-Net")
    ap.add_argument("--only_fold", type=int, default=-1, help="run a single fold (smoke test)")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seeds = [int(s) for s in args.seeds.split(",")]
    n_folds = len(json.load(open(args.folds_json))["folds"])
    folds = range(n_folds) if args.only_fold < 0 else [args.only_fold]

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.pred_csv).parent.mkdir(parents=True, exist_ok=True)

    # CSV (append-safe header)
    csv_fields = ["model", "seed", "fold", "patient_id", "label",
                  "prob_malignant", "predicted", "uncertainty"]
    write_header = not Path(args.pred_csv).exists()
    csv_fh = open(args.pred_csv, "a", newline="")
    writer = csv.DictWriter(csv_fh, fieldnames=csv_fields)
    if write_header:
        writer.writeheader()

    summary = []  # (seed, fold, f1, auc, acc)
    t0 = time.time()

    for seed in seeds:
        for fold in folds:
            print(f"\n{'#'*60}\n# seed={seed} fold={fold}\n{'#'*60}")
            set_seed(seed)

            loaders, class_weights = get_cv_dataloaders(
                args.data_root, args.folds_json, fold,
                batch_size=args.batch_size, oversample=True)

            # train_single_model expects dataloaders dict with train/val/test/class_weights.
            # For CV, the fold's val set is BOTH the early-stopping signal and the eval set.
            dl = {
                "train": loaders["train"],
                "val": loaders["val"],
                "test": loaders["val"],          # CV: evaluate on the fold val
                "class_weights": class_weights,
            }

            # per-(seed,fold) output dir so checkpoints don't collide
            run_args = argparse.Namespace(**vars(args))
            run_args.output_dir = str(Path(args.output_dir) / f"fold_{fold}")
            run_args.epochs = args.epochs
            run_args.patience = args.patience

            model, fold_metrics = train_single_model(run_args, seed, dl, device)

            # log per-sample val predictions
            criterion = CUEDNetLoss(num_classes=2)
            rows = collect_val_predictions(model, loaders["val"], device,
                                           criterion, class_weights)
            for r in rows:
                r.update({"model": args.model_name, "seed": seed, "fold": fold})
                writer.writerow(r)
            csv_fh.flush()

            summary.append((seed, fold, fold_metrics["f1"], fold_metrics["auc"],
                            fold_metrics.get("accuracy", float("nan"))))
            print(f"[cv] seed={seed} fold={fold} -> "
                  f"F1={fold_metrics['f1']:.4f} AUC={fold_metrics['auc']:.4f}")

            del model
            torch.cuda.empty_cache()

    csv_fh.close()

    # ---- aggregate ----
    arr = np.array([(s, f, f1, au) for (s, f, f1, au, _) in summary], dtype=float)
    out = {"model": args.model_name, "per_run": [
        {"seed": int(s), "fold": int(f), "f1": f1, "auc": au, "acc": ac}
        for (s, f, f1, au, ac) in summary]}

    # per-seed mean over folds, then mean±std across seeds
    per_seed_f1, per_seed_auc = [], []
    for seed in seeds:
        m = arr[arr[:, 0] == seed]
        if len(m):
            per_seed_f1.append(m[:, 2].mean())
            per_seed_auc.append(m[:, 3].mean())
    if per_seed_f1:
        out["cv_summary"] = {
            "f1_mean": float(np.mean(per_seed_f1)),
            "f1_std": float(np.std(per_seed_f1, ddof=1)) if len(per_seed_f1) > 1 else 0.0,
            "auc_mean": float(np.mean(per_seed_auc)),
            "auc_std": float(np.std(per_seed_auc, ddof=1)) if len(per_seed_auc) > 1 else 0.0,
            "n_seeds": len(per_seed_f1), "n_folds": len(folds),
        }
        print(f"\n[cv] {args.model_name} 5x5: "
              f"F1={out['cv_summary']['f1_mean']:.4f}±{out['cv_summary']['f1_std']:.4f}  "
              f"AUC={out['cv_summary']['auc_mean']:.4f}±{out['cv_summary']['auc_std']:.4f}")

    out["minutes"] = (time.time() - t0) / 60
    json.dump(out, open(Path(args.output_dir) / "cv_results.json", "w"), indent=2)
    print(f"[cv] preds -> {args.pred_csv}")
    print(f"[cv] summary -> {Path(args.output_dir)/'cv_results.json'}  ({out['minutes']:.1f} min)")


if __name__ == "__main__":
    main()
