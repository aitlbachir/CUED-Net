#!/usr/bin/env python
"""Learning-curve experiment over training-set fractions."""

import argparse, csv, json, sys, time, functools
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler

sys.path.insert(0, "/workspace/cued_net")

from models.cued_net import CUEDNetLoss
import train_cued_net
from train_cued_net import train_single_model, set_seed
import cv_dataloaders
from cv_dataloaders import get_cv_dataloaders

SEEDS = [42, 123]
FRACTIONS = [0.10, 0.25, 0.50, 0.75, 1.00]
BASE_EPOCHS = 40          # full-data convergence point; scaled up for smaller fracs
FIXED_PATIENCE = 15
N_FOLDS_DEFAULT = 5

DATA_ROOT = "/workspace/cbis-ddsm"
FOLDS_JSON = "/workspace/cued_net/cv_folds.json"
ROOT = Path("/workspace/cued_net/cv_learning_curve")


# ─────────────────────────────────────────────────────────────────────────────
def stratified_subsample(train_pos, labels, fraction, rng):
    """Subsample train_pos to `fraction`, preserving the per-class ratio.

    train_pos : list[int] dataset indices for this fold's training set
    labels    : list[int] parallel labels (same order as train_pos)
    Returns the kept subset of train_pos (list[int]).
    """
    if fraction >= 1.0:
        return list(train_pos)
    by_class = defaultdict(list)
    for idx, lab in zip(train_pos, labels):
        by_class[lab].append(idx)
    kept = []
    for lab, idxs in by_class.items():
        idxs = np.array(idxs)
        n_keep = max(1, int(round(len(idxs) * fraction)))
        sel = rng.choice(len(idxs), size=n_keep, replace=False)
        kept.extend(idxs[sel].tolist())
    rng.shuffle(kept)
    return kept


def build_subset_train_loader(ds_train_view, kept_pos, batch_size=16, num_workers=4):
    """Rebuild the train loader on `kept_pos`, mirroring cv_dataloaders.py
    lines 127-141 (Subset + WeightedRandomSampler), with class weights
    recomputed from the SUBSET labels."""
    train_subset = Subset(ds_train_view, kept_pos)
    labels = [ds_train_view.pairs[i]["label"] for i in kept_pos]
    counts = Counter(labels)
    weights = [1.0 / counts[l] for l in labels]
    sampler = WeightedRandomSampler(weights, len(weights) * 2, replacement=True)
    train_loader = DataLoader(train_subset, batch_size=batch_size, sampler=sampler,
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    bc = np.bincount(labels, minlength=2)
    cw = len(labels) / (2 * bc + 1e-6)
    class_weights = torch.tensor(cw, dtype=torch.float32)
    return train_loader, class_weights, bc


def collect_val_predictions(model, loader, device):
    """Identical schema to run_backbone_ablation/run_ablations."""
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


# ─────────────────────────────────────────────────────────────────────────────
def run(smoke):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # PATCH loss -> FINAL no_vdl config (lambda_vdl=0), same as backbone harness.
    def _patched_loss(num_classes=2, lambda_vdl=0.3, lambda_kl=0.1):
        return CUEDNetLoss(num_classes=num_classes, lambda_vdl=0.0, lambda_kl=lambda_kl)
    train_cued_net.CUEDNetLoss = _patched_loss
    print("[patch] CUEDNetLoss -> lambda_vdl=0.0 (final config)")

    n_folds = len(json.load(open(FOLDS_JSON))["folds"])
    seeds = [42] if smoke else SEEDS
    fractions = [1.00] if smoke else FRACTIONS
    folds = [0] if smoke else list(range(n_folds))

    ROOT.mkdir(parents=True, exist_ok=True)
    csv_fields = ["model","fraction","seed","fold","patient_id","label",
                  "prob_malignant","predicted","uncertainty"]

    summary = []
    t0 = time.time()

    for frac in fractions:
        pct = int(round(frac * 100))
        pred_csv = ROOT / f"frac_{pct}_preds.csv"
        write_header = not pred_csv.exists()
        fh = open(pred_csv, "a", newline="")
        writer = csv.DictWriter(fh, fieldnames=csv_fields)
        if write_header: writer.writeheader()

        # Option 2: epochs scale inversely with fraction (fixed training effort)
        epochs = max(BASE_EPOCHS, int(round(BASE_EPOCHS / frac)))

        for seed in seeds:
            for fold in folds:
                print(f"\n{'#'*60}")
                print(f"# frac={pct}% seed={seed} fold={fold} epochs={epochs} "
                      f"(Option2 fixed-effort)")
                print(f"{'#'*60}")
                set_seed(seed)

                # full fold (reuses ALL provenance gates) — we take ds + train_pos
                loaders, _cw = get_cv_dataloaders(
                    DATA_ROOT, FOLDS_JSON, fold, batch_size=16, oversample=True)

                # recover ds_train_view + train_pos by rebuilding the same resolution.
                # cv_dataloaders exposes neither directly, so reconstruct via its
                # internals: the full-cohort dataset + fold train_idx resolution.
                folds_rec = json.load(open(FOLDS_JSON))
                fold_rec = folds_rec["folds"][fold]
                fpairs = folds_rec["pairs"]
                ds_train_view = cv_dataloaders._build_full_cohort_dataset(
                    DATA_ROOT, cv_dataloaders._train_transform(224))
                ds_pos = {cv_dataloaders._content_key(p): i
                          for i, p in enumerate(ds_train_view.pairs)}
                train_pos = []
                for fi in fold_rec["train_idx"]:
                    k = cv_dataloaders._content_key(fpairs[fi])
                    train_pos.append(ds_pos[k])
                train_labels = [ds_train_view.pairs[i]["label"] for i in train_pos]

                # stratified subsample (seed-dependent so seeds differ)
                rng = np.random.default_rng(seed * 1000 + fold)
                kept = stratified_subsample(train_pos, train_labels, frac, rng)
                tr_loader, cw_sub, bc = build_subset_train_loader(
                    ds_train_view, kept, batch_size=16)
                print(f"[lc] frac={pct}%: kept {len(kept)}/{len(train_pos)} train "
                      f"({bc[1]} mal/{bc[0]} ben)")

                dl = {"train": tr_loader, "val": loaders["val"],
                      "test": loaders["val"], "class_weights": cw_sub}

                run_args = argparse.Namespace(
                    output_dir=str(ROOT / f"frac_{pct}" / f"seed_{seed}" / f"fold_{fold}"),
                    epochs=(3 if smoke else epochs),
                    patience=(2 if smoke else FIXED_PATIENCE),
                    lr=1e-4,
                )
                model, fold_metrics = train_single_model(run_args, seed, dl, device)

                rows = collect_val_predictions(model, loaders["val"], device)
                for r in rows:
                    r.update({"model": "CUED-Net", "fraction": pct,
                              "seed": seed, "fold": fold})
                    writer.writerow(r)
                fh.flush()

                summary.append((pct, seed, fold, fold_metrics["f1"], fold_metrics["auc"]))
                print(f"[lc] frac={pct}% seed={seed} fold={fold} -> "
                      f"F1={fold_metrics['f1']:.4f} AUC={fold_metrics['auc']:.4f}")
                del model
                if device.type == "cuda": torch.cuda.empty_cache()

        fh.close()

    # ── aggregate: per-fraction F1 mean + spread (the variance story) ───────
    arr = np.array(summary, dtype=float)  # cols: pct, seed, fold, f1, auc
    out = {"per_run": [{"fraction":int(p),"seed":int(s),"fold":int(f),
                        "f1":f1,"auc":au} for (p,s,f,f1,au) in summary]}
    if not smoke:
        per_frac = {}
        for pct in [int(round(f*100)) for f in FRACTIONS]:
            m = arr[arr[:,0]==pct]
            if len(m):
                per_frac[pct] = {
                    "f1_mean": float(m[:,3].mean()),
                    "f1_std":  float(m[:,3].std(ddof=1)),
                    "f1_min":  float(m[:,3].min()),
                    "f1_max":  float(m[:,3].max()),
                    "f1_range":float(m[:,3].max()-m[:,3].min()),
                    "auc_mean":float(m[:,4].mean()),
                    "auc_std": float(m[:,4].std(ddof=1)),
                    "n": int(len(m)),
                }
        out["per_fraction"] = per_frac
        print(f"\n{'='*64}\nLEARNING CURVE — per-fraction F1 spread")
        print(f"{'='*64}")
        print(f"{'frac':>6} {'n':>3} {'F1 mean':>9} {'F1 std':>8} "
              f"{'F1 min':>8} {'F1 max':>8} {'range':>8}")
        for pct in sorted(per_frac):
            d = per_frac[pct]
            print(f"{pct:>5}% {d['n']:>3} {d['f1_mean']:>9.4f} {d['f1_std']:>8.4f} "
                  f"{d['f1_min']:>8.4f} {d['f1_max']:>8.4f} {d['f1_range']:>8.4f}")
        print(f"{'='*64}")
        print("EXPECT: F1 std and range DECREASE as fraction grows -> supports "
              "'0.633-0.808 was a single-split artifact'.")

    out["minutes"] = (time.time()-t0)/60
    json.dump(out, open(ROOT / "learning_curve_results.json","w"), indent=2)
    print(f"\n[lc] results -> {ROOT/'learning_curve_results.json'} "
          f"({out['minutes']:.1f} min)")

    if smoke:
        pct, seed, fold, f1, auc = summary[0]
        print(f"\n{'='*60}\nSMOKE (frac={pct}%, seed={seed}, fold={fold}, 3 epochs)")
        print(f"  F1={f1:.4f}  AUC={auc:.4f}")
        print(f"  Sanity: AUC in band -> {'PASS' if 0.55<=auc<=0.97 else 'CHECK'}")
        print(f"  Subsample + loader-rebuild ran without error.")
        print(f"{'='*60}")


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--smoke", action="store_true")
    g.add_argument("--full", action="store_true")
    args = ap.parse_args()
    run(smoke=args.smoke)


if __name__ == "__main__":
    main()
