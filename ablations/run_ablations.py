#!/usr/bin/env python
"""
run_ablations.py — Component ablation harness for CUED-Net (R2.7).

Arms:
  no_vdl          : lambda_vdl=0.0, consistency kept (0.1)
  no_consistency  : lambda_vdl=0.3, consistency=0.0
  (full model is the existing cv_cued_net run; not re-run here)

SAFETY:
  - Does NOT edit train_cued_net.py or train_cv.py (locked, reproducible).
  - Monkeypatches train_cued_net.CUEDNetLoss in-process only.
  - Writes to ISOLATED dirs: cv_ablation/<arm>/...  and  cv_ablation/<arm>_preds.csv
  - Never touches cv_cued_net/ or cv_preds/.

USAGE:
  # Smoke test (seed=42, fold=0) — verify before full launch
  python run_ablations.py --arm no_vdl --smoke

  # Full 5x5
  python run_ablations.py --arm no_vdl --full
  python run_ablations.py --arm no_consistency --full
"""
import argparse, csv, json, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "/workspace/cued_net")

# import the REAL loss to subclass, and the training machinery
from models.cued_net import CUEDNetLoss
import train_cued_net
from train_cued_net import train_single_model, set_seed
from cv_dataloaders import get_cv_dataloaders

SEEDS = [42, 123, 456, 789, 2024]

# ─────────────────────────────────────────────────────────────────────────────
# Ablation loss: inherits _evidential_loss + _kl_divergence UNCHANGED,
# overrides ONLY the forward() combination line.
# ─────────────────────────────────────────────────────────────────────────────
class CUEDNetLossAblation(CUEDNetLoss):
    def __init__(self, num_classes=2, lambda_vdl=0.3, lambda_kl=0.1,
                 lambda_consistency=0.1, annealing_epochs=10):
        super().__init__(num_classes=num_classes, lambda_vdl=lambda_vdl,
                         lambda_kl=lambda_kl, annealing_epochs=annealing_epochs)
        self.lambda_consistency = lambda_consistency

    def forward(self, outputs, targets, epoch=0, class_weights=None):
        cc_out = outputs['cc_out']
        mlo_out = outputs['mlo_out']

        # evidential (inherited, identical to locked runs)
        loss_cc = self._evidential_loss(cc_out['alpha'], targets, epoch, class_weights)
        loss_mlo = self._evidential_loss(mlo_out['alpha'], targets, epoch, class_weights)
        evidential_loss = loss_cc + loss_mlo

        # VDL term (identical formula to original)
        cc_pred = torch.argmax(cc_out['prob'], dim=1)
        mlo_pred = torch.argmax(mlo_out['prob'], dim=1)
        disagreement = (cc_pred != mlo_pred).float()
        cc_conf = 1.0 - cc_out['uncertainty']
        mlo_conf = 1.0 - mlo_out['uncertainty']
        vdl = disagreement * cc_conf * mlo_conf
        vdl_loss = vdl.mean()

        # consistency term (identical formula)
        consistency_loss = F.mse_loss(cc_out['prob'], mlo_out['prob'])

        anneal = min(1.0, epoch / self.annealing_epochs)

        # ONLY CHANGE: coefficients are now configurable
        total_loss = evidential_loss \
            + self.lambda_vdl * anneal * vdl_loss \
            + self.lambda_consistency * anneal * consistency_loss

        return {
            'total': total_loss,
            'evidential': evidential_loss,
            'vdl': vdl_loss,
            'consistency': consistency_loss,
        }

# arm configs
ARMS = {
    "no_vdl":         dict(lambda_vdl=0.0, lambda_consistency=0.1),
    "no_consistency": dict(lambda_vdl=0.3, lambda_consistency=0.0),
}

# ─────────────────────────────────────────────────────────────────────────────
def collect_val_predictions(model, loader, device):
    """Mirror train_cv.collect_val_predictions, plus discordance-signal stats."""
    model.eval()
    rows = []
    disc_vals, vdl_proxy = [], []
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
            # discordance signal for R2.7 loss-vs-signal analysis
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
def run(arm, smoke):
    cfg = ARMS[arm]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # monkeypatch the loss IN-PROCESS ONLY
    def _patched_loss(num_classes=2, lambda_vdl=0.3, lambda_kl=0.1):
        # train_single_model calls CUEDNetLoss(num_classes=2, lambda_vdl=0.3, lambda_kl=0.1);
        # we ignore its lambda_vdl and inject the arm config instead.
        return CUEDNetLossAblation(
            num_classes=num_classes,
            lambda_vdl=cfg["lambda_vdl"],
            lambda_kl=lambda_kl,
            lambda_consistency=cfg["lambda_consistency"],
        )
    train_cued_net.CUEDNetLoss = _patched_loss
    print(f"[patch] CUEDNetLoss -> ablation arm '{arm}' "
          f"(lambda_vdl={cfg['lambda_vdl']}, lambda_consistency={cfg['lambda_consistency']})")

    out_dir = Path(f"/workspace/cued_net/cv_ablation/{arm}")
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_csv = Path(f"/workspace/cued_net/cv_ablation/{arm}_preds.csv")

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

    model_name = f"CUED-Net-{arm}"
    summary, disc_log = [], []
    t0 = time.time()

    for seed in seeds:
        for fold in folds:
            print(f"\n{'#'*60}\n# arm={arm} seed={seed} fold={fold}\n{'#'*60}")
            set_seed(seed)
            loaders, class_weights = get_cv_dataloaders(
                data_root, folds_json, fold, batch_size=16, oversample=True)
            dl = {"train": loaders["train"], "val": loaders["val"],
                  "test": loaders["val"], "class_weights": class_weights}

            run_args = argparse.Namespace(
                output_dir=str(out_dir / f"fold_{fold}"),
                epochs=(3 if smoke else 50),     # smoke: 3 epochs only
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
            print(f"[abl:{arm}] seed={seed} fold={fold} -> "
                  f"F1={fold_metrics['f1']:.4f} AUC={fold_metrics['auc']:.4f}  "
                  f"disc_mean={disc_stats['disc_mean']}")
            del model
            if device.type == "cuda": torch.cuda.empty_cache()

    fh.close()

    arr = np.array([(s,f,f1,au) for (s,f,f1,au) in summary], dtype=float)
    out = {"arm": arm, "config": cfg,
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
        print(f"\n[abl:{arm}] 5x5: F1={out['cv_summary']['f1_mean']:.4f}"
              f"±{out['cv_summary']['f1_std']:.4f}  "
              f"AUC={out['cv_summary']['auc_mean']:.4f}±{out['cv_summary']['auc_std']:.4f}")
    out["minutes"] = (time.time()-t0)/60
    json.dump(out, open(out_dir / "ablation_results.json","w"), indent=2)
    print(f"[abl:{arm}] preds -> {pred_csv}")
    print(f"[abl:{arm}] summary -> {out_dir/'ablation_results.json'} ({out['minutes']:.1f} min)")

    if smoke:
        f1, auc = summary[0][2], summary[0][3]
        print(f"\n{'='*60}\nSMOKE TEST RESULT (arm={arm}, seed=42, fold=0, 3 epochs)")
        print(f"  F1={f1:.4f}  AUC={auc:.4f}")
        ok = (0.55 <= auc <= 0.97) and ("prob" )
        print(f"  Sanity: AUC in plausible band -> {'PASS' if 0.55<=auc<=0.97 else 'CHECK'}")
        print(f"  (3-epoch smoke F1/AUC will be LOWER than full 50-epoch; "
              f"we only check it trains & emits valid outputs.)")
        print(f"{'='*60}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=list(ARMS.keys()))
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--smoke", action="store_true")
    g.add_argument("--full", action="store_true")
    args = ap.parse_args()
    run(args.arm, smoke=args.smoke)

if __name__ == "__main__":
    main()