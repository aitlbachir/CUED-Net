#!/usr/bin/env python
"""Backbone comparison ablation."""

import argparse, csv, json, sys, time, functools
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, "/workspace/cued_net")

import models.cued_net as cued_mod
from models.cued_net import CUEDNetLoss
import train_cued_net
from train_cued_net import train_single_model, set_seed
from cv_dataloaders import get_cv_dataloaders

# backbone-swappable encoder (placed on pod next to this file)
from view_encoder_backbone import ViewEncoderBackbone

# Table III: 3 seeds (reduced-ablation budget, agreed). DenseNet-121 is the
# existing cv_cued_net/no_vdl result and is NOT re-run here.
SEEDS = [42, 123, 456]
BACKBONES = ["resnet50", "efficientnet_b0", "convnext_tiny"]


# ─────────────────────────────────────────────────────────────────────────────
def collect_val_predictions(model, loader, device):
    """Identical to run_ablations.collect_val_predictions."""
    model.eval()
    rows = []
    disc_vals = []
    with torch.no_grad():
        for batch in loader:
            cc = batch["img_cc"].to(device)
            mlo = batch["img_mlo"].to(device)
            labels = batch["label"].numpy()
            pids = batch.get("patient_id", ["?"] * len(labels))
            out = model(cc, mlo)
            prob_mal = out["prob"][:, 1].cpu().numpy()
            pred = (prob_mal >= 0.5).astype(int)
            u = out.get("uncertainty_combined", out.get("uncertainty_evidential"))
            u = u.cpu().numpy() if u is not None else np.full(len(labels), np.nan)
            d = out.get("uncertainty_discordance")
            if d is not None:
                disc_vals.extend(d.cpu().numpy().tolist())
            for i in range(len(labels)):
                rows.append({
                    "patient_id": pids[i] if isinstance(pids, (list, tuple)) else str(pids[i]),
                    "label": int(labels[i]),
                    "prob_malignant": float(prob_mal[i]),
                    "predicted": int(pred[i]),
                    "uncertainty": float(u[i]),
                })
    disc_stats = {
        "disc_mean": float(np.mean(disc_vals)) if disc_vals else None,
        "disc_std":  float(np.std(disc_vals)) if disc_vals else None,
    }
    return rows, disc_stats


# ─────────────────────────────────────────────────────────────────────────────
def run(backbone, smoke):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # (1) PATCH ViewEncoder -> backbone-swapped variant, IN-PROCESS ONLY.
    # CUEDNet.__init__ calls ViewEncoder(num_classes, pretrained); we bind the
    # chosen backbone via functools.partial so the call signature is preserved.
    _orig_encoder = cued_mod.ViewEncoder
    cued_mod.ViewEncoder = functools.partial(ViewEncoderBackbone, backbone=backbone)
    print(f"[patch] ViewEncoder -> ViewEncoderBackbone(backbone='{backbone}')")

    # (2) PATCH loss -> FINAL no_vdl config (lambda_vdl=0). train_single_model
    # hardcodes CUEDNetLoss(num_classes=2, lambda_vdl=0.3, lambda_kl=0.1).
    def _patched_loss(num_classes=2, lambda_vdl=0.3, lambda_kl=0.1):
        return CUEDNetLoss(num_classes=num_classes, lambda_vdl=0.0, lambda_kl=lambda_kl)
    train_cued_net.CUEDNetLoss = _patched_loss
    print(f"[patch] CUEDNetLoss -> lambda_vdl=0.0 (final no_vdl config)")

    out_dir = Path(f"/workspace/cued_net/cv_backbone/{backbone}")
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_csv = Path(f"/workspace/cued_net/cv_backbone/{backbone}_preds.csv")

    data_root = "/workspace/cbis-ddsm"
    folds_json = "/workspace/cued_net/cv_folds.json"
    n_folds = len(json.load(open(folds_json))["folds"])

    seeds = [42] if smoke else SEEDS
    folds = [0] if smoke else list(range(n_folds))

    csv_fields = ["model","seed","fold","patient_id","label",
                  "prob_malignant","predicted","uncertainty"]
    write_header = not pred_csv.exists()
    fh = open(pred_csv, "a", newline="")
    writer = csv.DictWriter(fh, fieldnames=csv_fields)
    if write_header: writer.writeheader()

    model_name = f"CUED-Net-{backbone}"
    summary, disc_log = [], []
    t0 = time.time()

    for seed in seeds:
        for fold in folds:
            print(f"\n{'#'*60}\n# backbone={backbone} seed={seed} fold={fold}\n{'#'*60}")
            set_seed(seed)
            loaders, class_weights = get_cv_dataloaders(
                data_root, folds_json, fold, batch_size=16, oversample=True)
            dl = {"train": loaders["train"], "val": loaders["val"],
                  "test": loaders["val"], "class_weights": class_weights}

            run_args = argparse.Namespace(
                output_dir=str(out_dir / f"fold_{fold}"),
                epochs=(3 if smoke else 50),
                patience=(2 if smoke else 15),
                lr=1e-4,
            )
            model, fold_metrics = train_single_model(run_args, seed, dl, device)

            rows, disc_stats = collect_val_predictions(model, loaders["val"], device)
            for r in rows:
                r.update({"model": model_name, "seed": seed, "fold": fold})
                writer.writerow(r)
            fh.flush()
            disc_log.append({"seed": seed, "fold": fold, **disc_stats})

            summary.append((seed, fold, fold_metrics["f1"], fold_metrics["auc"]))
            print(f"[bb:{backbone}] seed={seed} fold={fold} -> "
                  f"F1={fold_metrics['f1']:.4f} AUC={fold_metrics['auc']:.4f}  "
                  f"disc_mean={disc_stats['disc_mean']}")
            del model
            if device.type == "cuda": torch.cuda.empty_cache()

    fh.close()

    # restore patches (hygiene; process usually exits anyway)
    cued_mod.ViewEncoder = _orig_encoder

    arr = np.array([(s,f,f1,au) for (s,f,f1,au) in summary], dtype=float)
    out = {"backbone": backbone,
           "per_run": [{"seed":int(s),"fold":int(f),"f1":f1,"auc":au}
                       for (s,f,f1,au) in summary],
           "disc_signal": disc_log}
    if not smoke and len(arr):
        per_seed_f1, per_seed_auc = [], []
        for seed in seeds:
            m = arr[arr[:,0]==seed]
            if len(m):
                per_seed_f1.append(m[:,2].mean()); per_seed_auc.append(m[:,3].mean())
        out["cv_summary"] = {
            "f1_mean": float(np.mean(per_seed_f1)),
            "f1_std":  float(np.std(per_seed_f1, ddof=1)),
            "auc_mean":float(np.mean(per_seed_auc)),
            "auc_std": float(np.std(per_seed_auc, ddof=1)),
        }
        print(f"\n[bb:{backbone}] 3x5: F1={out['cv_summary']['f1_mean']:.4f}"
              f"±{out['cv_summary']['f1_std']:.4f}  "
              f"AUC={out['cv_summary']['auc_mean']:.4f}±{out['cv_summary']['auc_std']:.4f}")
    out["minutes"] = (time.time()-t0)/60
    json.dump(out, open(out_dir / "backbone_results.json","w"), indent=2)
    print(f"[bb:{backbone}] preds -> {pred_csv}")
    print(f"[bb:{backbone}] summary -> {out_dir/'backbone_results.json'} ({out['minutes']:.1f} min)")

    if smoke:
        f1, auc = summary[0][2], summary[0][3]
        print(f"\n{'='*60}\nSMOKE TEST RESULT (backbone={backbone}, seed=42, fold=0, 3 epochs)")
        print(f"  F1={f1:.4f}  AUC={auc:.4f}")
        print(f"  Sanity: AUC in plausible band -> {'PASS' if 0.55<=auc<=0.97 else 'CHECK'}")
        print(f"  (3-epoch smoke F1/AUC will be LOWER than full 50-epoch; "
              f"we only check it trains & emits valid outputs.)")
        print(f"{'='*60}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", required=True, choices=BACKBONES)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--smoke", action="store_true")
    g.add_argument("--full", action="store_true")
    args = ap.parse_args()
    run(args.backbone, smoke=args.smoke)

if __name__ == "__main__":
    main()
