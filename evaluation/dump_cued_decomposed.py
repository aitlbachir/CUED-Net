#!/usr/bin/env python3
"""Dump per-sample decomposed uncertainty from trained checkpoints."""

import argparse, glob, json, os, sys
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, "/workspace/cued_net")
sys.path.insert(0, "/workspace/cued_net/models")

from models.cued_net import CUEDNet, CUEDNetEnsemble   # noqa: E402
import cv_dataloaders                                   # noqa: E402

SEEDS = [42, 123, 456, 789, 2024]
FOLDS = [0, 1, 2, 3, 4]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def ckpt_path(root, seed, fold):
    return os.path.join(root, f"fold_{fold}", f"seed_{seed}", "best_model.pt")


def load_model(path):
    ck = torch.load(path, map_location=DEVICE, weights_only=False)
    state = ck.get("model_state_dict", ck) if isinstance(ck, dict) else ck
    net = CUEDNet(num_classes=2, pretrained=False)
    net.load_state_dict(state)
    net.to(DEVICE).eval()
    return net


def build_val_loader(data_root, folds_json, fold, batch_size):
    """Reproduce train_cv.py's call exactly. loaders is a dict; we need 'val'."""
    loaders, _cw = cv_dataloaders.get_cv_dataloaders(
        data_root, folds_json, fold, batch_size=batch_size, oversample=True)
    return loaders["val"]


@torch.no_grad()
def run_single_model(net, val_loader):
    """Mirror collect_val_predictions: out = model(cc, mlo); prob[:,1];
    uncertainty_combined. Also pull u_evid and u_disc from the same forward."""
    rows = {k: [] for k in ("patient_id","label","p_mal","pred",
                            "u_combined","u_evid","u_disc")}
    for batch in val_loader:
        cc = batch["img_cc"].to(DEVICE)
        mlo = batch["img_mlo"].to(DEVICE)
        labels = batch["label"]
        pids = batch.get("patient_id", ["?"] * len(labels))
        out = net(cc, mlo)
        p_mal = out["prob"][:, 1].detach().cpu().numpy()
        pred = (p_mal >= 0.5).astype(int)
        u_comb = out["uncertainty_combined"].detach().cpu().numpy().reshape(-1)
        u_evid = out["uncertainty_evidential"].detach().cpu().numpy().reshape(-1)
        u_disc = out["uncertainty_discordance"].detach().cpu().numpy().reshape(-1)
        labs = (labels.detach().cpu().numpy().reshape(-1)
                if torch.is_tensor(labels) else np.asarray(labels).reshape(-1))
        for i in range(len(p_mal)):
            rows["patient_id"].append(str(pids[i]))
            rows["label"].append(int(labs[i]))
            rows["p_mal"].append(float(p_mal[i]))
            rows["pred"].append(int(pred[i]))
            rows["u_combined"].append(float(u_comb[i]))
            rows["u_evid"].append(float(u_evid[i]))
            rows["u_disc"].append(float(u_disc[i]))
    return rows


@torch.no_grad()
def run_ensemble_fold(models_for_fold, val_loader, _printed=[False]):
    """Proper CV ensemble for ONE fold: the 5 seed-models trained on this fold,
    ensembled, evaluated on this fold's held-out val. CUEDNetEnsemble.predict
    keys (verified): prob, uncertainty_evidential, uncertainty_ensemble,
    uncertainty_discordance, uncertainty_total, view_agreement."""
    ens = CUEDNetEnsemble(models_for_fold)
    rows = {k: [] for k in ("patient_id","label","p_mal","pred",
                            "u_evid","u_ens","u_disc","u_total")}
    for batch in val_loader:
        cc = batch["img_cc"].to(DEVICE)
        mlo = batch["img_mlo"].to(DEVICE)
        labels = batch["label"]
        pids = batch.get("patient_id", ["?"] * len(labels))
        out = ens.predict(cc, mlo, DEVICE)
        if not _printed[0]:
            print(f"\n    [ensemble predict() keys: {list(out.keys())}]")
            _printed[0] = True
        prob = out["prob"].detach().cpu().numpy()
        p_mal = prob[:, 1] if (prob.ndim == 2 and prob.shape[1] == 2) else prob.reshape(-1)
        pred = (p_mal >= 0.5).astype(int)
        n = len(p_mal)
        def g(*names, default=None):
            for nm in names:
                if nm in out and out[nm] is not None:
                    return out[nm].detach().cpu().numpy().reshape(-1)
            if default is not None:
                return np.full(n, default, dtype=float)
            raise KeyError(f"none of {names} in {list(out.keys())}")
        u_evid = g("uncertainty_evidential")
        u_ens  = g("uncertainty_ensemble", default=0.0)
        u_disc = g("uncertainty_discordance")
        u_total = g("uncertainty_total", "uncertainty_combined")
        labs = (labels.detach().cpu().numpy().reshape(-1)
                if torch.is_tensor(labels) else np.asarray(labels).reshape(-1))
        for i in range(n):
            rows["patient_id"].append(str(pids[i]))
            rows["label"].append(int(labs[i]))
            rows["p_mal"].append(float(p_mal[i]))
            rows["pred"].append(int(pred[i]))
            rows["u_evid"].append(float(u_evid[i]))
            rows["u_ens"].append(float(u_ens[i]))
            rows["u_disc"].append(float(u_disc[i]))
            rows["u_total"].append(float(u_total[i]))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cv_ckpt_root", default="/workspace/cued_net/cv_cued_net",
                    help="root containing fold_{F}/seed_{S}/best_model.pt")
    ap.add_argument("--data_root", default="/workspace/cbis-ddsm")
    ap.add_argument("--folds_json", default="/workspace/cued_net/cv_folds.json")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--out_dir", default="/workspace/cued_net/cv_preds")
    args = ap.parse_args()

    print(f"[+] Device: {DEVICE}")

    # Verify all 25 checkpoints exist up front.
    missing = []
    for s in SEEDS:
        for f in FOLDS:
            if not os.path.exists(ckpt_path(args.cv_ckpt_root, s, f)):
                missing.append((s, f))
    if missing:
        raise SystemExit(f"[error] missing CV checkpoints for {missing}")
    print(f"[+] all 25 CV checkpoints present under {args.cv_ckpt_root}")

    # Build each fold's val loader ONCE (deterministic across seeds).
    val_loaders = {}
    for fold in FOLDS:
        print(f"[+] building val loader for fold {fold} …")
        val_loaders[fold] = build_val_loader(
            args.data_root, args.folds_json, fold, args.batch_size)

    # ---- (1) PER-SEED: load (seed,fold) model, eval on fold's val ----
    per_seed_frames = []
    for seed in SEEDS:
        for fold in FOLDS:
            print(f"[per-seed] seed={seed} fold={fold} …", end=" ", flush=True)
            net = load_model(ckpt_path(args.cv_ckpt_root, seed, fold))
            r = run_single_model(net, val_loaders[fold])
            df = pd.DataFrame({
                "model": "CUED-Net", "seed": seed, "fold": fold,
                "patient_id": r["patient_id"], "label": r["label"],
                "prob_malignant": r["p_mal"], "predicted": r["pred"],
                "uncertainty": r["u_combined"],            # = legacy column
                "uncertainty_combined": r["u_combined"],
                "uncertainty_evidential": r["u_evid"],
                "uncertainty_discordance": r["u_disc"],
            })
            per_seed_frames.append(df)
            del net
            if DEVICE == "cuda":
                torch.cuda.empty_cache()
            print(f"{len(df)} rows")
    ps = pd.concat(per_seed_frames, ignore_index=True)
    ps = ps[["model","seed","fold","patient_id","label","prob_malignant",
             "predicted","uncertainty","uncertainty_combined",
             "uncertainty_evidential","uncertainty_discordance"]]
    out1 = os.path.join(args.out_dir, "cued_net_preds_decomposed.csv")
    ps.to_csv(out1, index=False)
    print(f"\n[ok] per-seed -> {out1}  ({len(ps)} rows; expect 2650)")

    # ---- (2) ENSEMBLE: per fold, ensemble the 5 seed-models for that fold ----
    ens_frames = []
    for fold in FOLDS:
        print(f"[ensemble] fold={fold} …", end=" ", flush=True)
        models_for_fold = [load_model(ckpt_path(args.cv_ckpt_root, s, fold))
                           for s in SEEDS]
        r = run_ensemble_fold(models_for_fold, val_loaders[fold])
        df = pd.DataFrame({
            "model": "CUED-Net-Ensemble", "seed": -1, "fold": fold,
            "patient_id": r["patient_id"], "label": r["label"],
            "prob_malignant": r["p_mal"], "predicted": r["pred"],
            "uncertainty": r["u_total"],
            "uncertainty_evidential": r["u_evid"],
            "uncertainty_ensemble": r["u_ens"],
            "uncertainty_discordance": r["u_disc"],
            "uncertainty_total": r["u_total"],
        })
        ens_frames.append(df)
        del models_for_fold
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        print(f"{len(df)} rows")
    en = pd.concat(ens_frames, ignore_index=True)
    en = en[["model","seed","fold","patient_id","label","prob_malignant",
             "predicted","uncertainty","uncertainty_evidential",
             "uncertainty_ensemble","uncertainty_discordance","uncertainty_total"]]
    out2 = os.path.join(args.out_dir, "cued_net_ensemble_preds.csv")
    en.to_csv(out2, index=False)
    print(f"[ok] ensemble -> {out2}  ({len(en)} rows; expect 530)")

    # ---- alignment + reproduction gate vs the OLD per-seed CSV ----
    old = os.path.join(args.out_dir, "cued_net_preds.csv")
    if os.path.exists(old):
        odf = pd.read_csv(old)
        same_n = len(odf) == len(ps)
        ok_order, max_dprob, max_dunc = True, 0.0, 0.0
        for (s, f), g in ps.groupby(["seed", "fold"]):
            og = odf[(odf.seed == s) & (odf.fold == f)].reset_index(drop=True)
            g = g.reset_index(drop=True)
            if len(g) != len(og) or list(g.patient_id) != list(og.patient_id):
                ok_order = False
                break
            max_dprob = max(max_dprob, float(np.abs(
                g.prob_malignant.values - og.prob_malignant.values).max()))
            if "uncertainty" in og.columns:
                max_dunc = max(max_dunc, float(np.abs(
                    g.uncertainty.values - og.uncertainty.values).max()))
        print(f"\n[check] row count matches old CSV: {same_n} "
              f"({len(odf)} old vs {len(ps)} new)")
        print(f"[check] per-cell patient_id ORDER matches old CSV: {ok_order}")
        if same_n and ok_order:
            print(f"[check] max |Δprob|        vs old: {max_dprob:.4g} (want ~0)")
            print(f"[check] max |Δuncertainty| vs old: {max_dunc:.4g} (want ~0)")
            if max_dprob < 1e-3 and max_dunc < 1e-3:
                print("[ok] VERIFIED: dump reproduces the original CV predictions.")
            else:
                print("[!] Δ exceeds tolerance — inspect before trusting the dump.")
        else:
            print("[!] Alignment differs — resolve before pooling.")


if __name__ == "__main__":
    main()
