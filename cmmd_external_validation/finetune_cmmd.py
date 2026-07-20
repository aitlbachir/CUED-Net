#!/usr/bin/env python
"""
finetune_cmmd.py — Light domain adaptation of CUED-Net to CMMD.

DESIGN (the part reviewers will scrutinize)
-------------------------------------------
1. PATIENT-LEVEL DISJOINT SPLIT. Train / val / test are split by patient_id,
   never by pair. A patient's CC+MLO pairs all land in exactly one split. This
   is the same leakage guarantee used for the CBIS 5-fold CV. We expose the
   exact patient lists used, so the split is auditable.

2. FROZEN ENCODERS. The two DenseNet-121 encoders are frozen. Only the
   evidential heads (and, optionally, the fusion parameters) are trained. This
   is "light fine-tuning": we adapt the decision/calibration layers to the new
   input distribution rather than relearning features from a small sample. It
   is the honest answer to "does it generalize with minimal target data".

3. SMALL TARGET FRACTION. By default we fine-tune on 10% of CMMD patients and
   evaluate on a held-out 70% test set (20% val for early stopping). The test
   set is touched only once, at the end. No threshold or hyperparameter is
   selected on the test labels.

4. SAME UQ EVALUATION as evaluate_cmmd.py is reproduced on the held-out test
   set so the fine-tuned and zero-shot results are directly comparable.

USAGE
-----
    python finetune_cmmd.py \
        --manifest    /workspace/cued_net/cmmd_pairs_full.json \
        --ckpt        /workspace/outputs_cued/seed_42/best_model.pt \
        --models_dir  /workspace/cued_net \
        --train_frac  0.10 --val_frac 0.20 \
        --epochs 30 --lr 1e-4 --batch_size 16 --seed 42 \
        --out_dir     /workspace/cued_net/finetune_out

Run across the same 5 seeds to report mean ± std, mirroring the CBIS protocol.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import roc_auc_score


# --------------------------------------------------------------------------- #
# Patient-level split (label-aware only for stratification, never for the test
# decision boundary)
# --------------------------------------------------------------------------- #
def patient_level_split(pairs, train_frac, val_frac, seed):
    """Split pair indices into train/val/test by patient_id.

    Stratify by per-patient max label (any malignant pair => patient positive),
    matching the CBIS CV stratification rule. This only controls *which
    patients* go to each split, never the per-sample classification threshold.
    """
    by_patient = {}
    for idx, p in enumerate(pairs):
        by_patient.setdefault(p["patient_id"], []).append(idx)

    patients = list(by_patient.keys())
    # per-patient label for stratified assignment
    plabel = {
        pid: max(int(pairs[i]["label"]) for i in idxs)
        for pid, idxs in by_patient.items()
    }

    rng = np.random.default_rng(seed)
    pos = [p for p in patients if plabel[p] == 1]
    neg = [p for p in patients if plabel[p] == 0]
    rng.shuffle(pos)
    rng.shuffle(neg)

    def cut(lst):
        n = len(lst)
        n_tr = int(round(n * train_frac))
        n_va = int(round(n * val_frac))
        return lst[:n_tr], lst[n_tr:n_tr + n_va], lst[n_tr + n_va:]

    pos_tr, pos_va, pos_te = cut(pos)
    neg_tr, neg_va, neg_te = cut(neg)

    tr_pat = set(pos_tr + neg_tr)
    va_pat = set(pos_va + neg_va)
    te_pat = set(pos_te + neg_te)

    # sanity: disjoint
    assert tr_pat.isdisjoint(va_pat) and tr_pat.isdisjoint(te_pat) and va_pat.isdisjoint(te_pat)

    def idxs_for(patset):
        out = []
        for pid in patset:
            out.extend(by_patient[pid])
        return sorted(out)

    return idxs_for(tr_pat), idxs_for(va_pat), idxs_for(te_pat), \
           {"train": sorted(tr_pat), "val": sorted(va_pat), "test": sorted(te_pat)}


# --------------------------------------------------------------------------- #
# Freeze encoders, expose heads + fusion
# --------------------------------------------------------------------------- #
def configure_trainable(model, train_fusion=True, unfreeze_denseblock4=False):
    """Freeze DenseNet backbones; train classifier + evidential heads.
    If unfreeze_denseblock4=True, ALSO unfreeze the last dense block
    (features.denseblock4) and final norm (features.norm5) in both encoders,
    for partial feature adaptation. Earlier blocks stay frozen.

    Returns (model, head_params, backbone_params) so the optimizer can apply
    a lower LR to the unfrozen backbone block (discriminative fine-tuning)."""
    head_params, backbone_params = [], []
    n_head, n_bb, n_froze = 0, 0, 0

    for name, param in model.named_parameters():
        is_backbone_feat = ".features." in name or name.endswith(".features")
        is_head = (".classifier" in name) or (".evidential" in name)
        # the deep block we optionally adapt
        is_db4 = ("features.denseblock4" in name) or ("features.norm5" in name)

        if is_head and not is_backbone_feat:
            param.requires_grad = True
            head_params.append(param); n_head += param.numel()
        elif unfreeze_denseblock4 and is_db4:
            param.requires_grad = True
            backbone_params.append(param); n_bb += param.numel()
        else:
            param.requires_grad = False
            n_froze += param.numel()

    print(f"[freeze] head params: {n_head:,} | denseblock4 params: {n_bb:,} | frozen: {n_froze:,}")
    if n_head == 0:
        print("[freeze][WARN] no head params matched — inspect named_parameters().")
    return model, head_params, backbone_params

def make_loss(models_dir):
    try:
        sys.path.insert(0, models_dir)
        from models.cued_net import CUEDNetLoss
        print("[loss] using CUEDNetLoss from cued_net.py")
        return CUEDNetLoss(), True
    except Exception as e:
        print(f"[loss] CUEDNetLoss unavailable ({e}); falling back to CE on prob.")
        return nn.NLLLoss(), False


# --------------------------------------------------------------------------- #
# Evaluation block — mirrors evaluate_cmmd.py metrics on a given subset
# --------------------------------------------------------------------------- #
def evaluate(model, loader, device):
    model.eval()
    probs, labels, u_total = [], [], []
    with torch.no_grad():
        for batch in loader:
            cc = batch["img_cc"].to(device)
            mlo = batch["img_mlo"].to(device)
            out = model(cc, mlo)
            probs.append(out["prob"][:, 1].cpu().numpy())
            labels.append(batch["label"].numpy())
            # uncertainty_combined is the fused total; fall back if absent
            u = out.get("uncertainty_combined", out.get("uncertainty_evidential"))
            u_total.append(u.cpu().numpy())
    probs = np.concatenate(probs)
    labels = np.concatenate(labels)
    u_total = np.concatenate(u_total)

    auc = roc_auc_score(labels, probs)

    # bootstrap CI
    rng = np.random.default_rng(0)
    boots = []
    for _ in range(1000):
        idx = rng.integers(0, len(probs), len(probs))
        if len(np.unique(labels[idx])) < 2:
            continue
        boots.append(roc_auc_score(labels[idx], probs[idx]))
    ci = (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))

    # selective prediction by uncertainty
    sel = {}
    order = np.argsort(u_total)  # most-certain first
    for cov in (1.0, 0.7, 0.5):
        k = int(len(probs) * cov)
        keep = order[:k]
        if len(np.unique(labels[keep])) < 2:
            sel[f"coverage_{int(cov*100)}"] = None
            continue
        sel[f"coverage_{int(cov*100)}"] = {
            "coverage": cov,
            "n_kept": int(k),
            "auc": float(roc_auc_score(labels[keep], probs[keep])),
        }

    return {
        "auc": float(auc),
        "auc_ci_95": [round(ci[0], 4), round(ci[1], 4)],
        "n": int(len(probs)),
        "selective_prediction": sel,
    }


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    sys.path.insert(0, args.models_dir)
    from cmmd_pair_dataset import CMMDPairDataset, cmmd_collate_fn
    from models.cued_net import CUEDNet

    # ---- data ----
    manifest = json.load(open(args.manifest))
    pairs = manifest["pairs"] if isinstance(manifest, dict) and "pairs" in manifest else manifest

    tr_idx, va_idx, te_idx, splits = patient_level_split(
        pairs, args.train_frac, args.val_frac, args.seed
    )
    print(f"[split] patients  train={len(splits['train'])} "
          f"val={len(splits['val'])} test={len(splits['test'])}")
    print(f"[split] pairs     train={len(tr_idx)} val={len(va_idx)} test={len(te_idx)}")

    full_ds = CMMDPairDataset(args.manifest)
    tr_ds, va_ds, te_ds = Subset(full_ds, tr_idx), Subset(full_ds, va_idx), Subset(full_ds, te_idx)

    dl = lambda ds, sh: DataLoader(ds, batch_size=args.batch_size, shuffle=sh,
                                   collate_fn=cmmd_collate_fn, num_workers=2)
    tr_dl, va_dl, te_dl = dl(tr_ds, True), dl(va_ds, False), dl(te_ds, False)

    # ---- model ----
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model = CUEDNet(num_classes=2)
    model.load_state_dict(ck["model_state_dict"])
    model.to(device)
    _, head_params, backbone_params = configure_trainable(
        model, train_fusion=not args.freeze_fusion,
        unfreeze_denseblock4=args.unfreeze_denseblock4)

    criterion, evidential_loss = make_loss(args.models_dir)
    param_groups = [{"params": head_params, "lr": args.lr}]
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": args.lr * args.backbone_lr_mult})
        print(f"[optim] head LR={args.lr:g}  backbone LR={args.lr*args.backbone_lr_mult:g}")
    optim = torch.optim.Adam(param_groups, weight_decay=1e-5)

    # ---- zero-shot baseline on THIS test split (apples-to-apples) ----
    zs = evaluate(model, te_dl, device)
    print(f"[zero-shot on test split] AUC={zs['auc']:.4f} CI={zs['auc_ci_95']}")

    # ---- training loop with val-AUC early stopping ----
    best_val, best_state, patience, bad = -1, None, args.patience, 0
    for epoch in range(args.epochs):
        model.train()
        # keep frozen encoder BN in eval mode so its running stats don't drift
        for mod in model.modules():
            if isinstance(mod, nn.BatchNorm2d) and not any(
                p.requires_grad for p in mod.parameters()
            ):
                mod.eval()

        running = 0.0
        for batch in tr_dl:
            cc = batch["img_cc"].to(device)
            mlo = batch["img_mlo"].to(device)
            y = batch["label"].to(device)
            optim.zero_grad()
            out = model(cc, mlo)
            if evidential_loss:
                loss_dict = criterion(out, y, epoch)
                # CUEDNetLoss returns a dict; 'total' is the combined scalar objective
                loss = loss_dict["total"] if isinstance(loss_dict, dict) else loss_dict
            else:
                loss = criterion(torch.log(out["prob"] + 1e-8), y)
            loss.backward()
            optim.step()
            running += loss.item() * y.size(0)

        val = evaluate(model, va_dl, device)
        print(f"  epoch {epoch:02d}  loss={running/len(tr_idx):.4f}  val_AUC={val['auc']:.4f}")
        if val["auc"] > best_val:
            best_val = val["auc"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                print(f"  early stop at epoch {epoch} (best val_AUC={best_val:.4f})")
                break

    # ---- restore best, evaluate ONCE on held-out test ----
    if best_state is not None:
        model.load_state_dict(best_state)
    ft = evaluate(model, te_dl, device)
    print(f"[fine-tuned on test split] AUC={ft['auc']:.4f} CI={ft['auc_ci_95']}")

    # ---- save ----
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    result = {
        "seed": args.seed,
        "train_frac": args.train_frac,
        "val_frac": args.val_frac,
        "patient_splits": splits,
        "zero_shot_test": zs,
        "finetuned_test": ft,
        "best_val_auc": float(best_val),
    }
    out_path = os.path.join(args.out_dir, f"finetune_seed{args.seed}.json")
    json.dump(result, open(out_path, "w"), indent=2)
    torch.save({"model_state_dict": model.state_dict()},
               os.path.join(args.out_dir, f"cued_cmmd_ft_seed{args.seed}.pt"))
    print(f"[done] -> {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--models_dir", default="/workspace/cued_net")
    ap.add_argument("--train_frac", type=float, default=0.10)
    ap.add_argument("--val_frac", type=float, default=0.20)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--freeze_fusion", action="store_true",
                    help="if set, fusion params are frozen too (heads-only)")
    ap.add_argument("--unfreeze_denseblock4", action="store_true",
                    help="also unfreeze features.denseblock4 + norm5 for partial feature adaptation")
    ap.add_argument("--backbone_lr_mult", type=float, default=0.1,
                    help="backbone LR = lr * this (discriminative fine-tuning; default 0.1)")
    ap.add_argument("--out_dir", default="/workspace/cued_net/finetune_out")
    main(ap.parse_args())
