#!/usr/bin/env python
"""Cross-validation dataloaders with patient-grouped, class-balanced sampling."""

import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler

import sys
sys.path.insert(0, "/workspace/cued_net")
from data.datasets import CBISDDSMDataset
import torchvision.transforms as T


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def _content_key(pair):
    return (pair["patient_id"], str(pair["cc_path"]).split("/")[-1])


def _build_full_cohort_dataset(data_root, training_transform):
    """Return a CBISDDSMDataset holding the FULL 530-pair pairable cohort
    (bypassing the internal train/val split), with the given transform."""
    ds = CBISDDSMDataset(data_root, split="train", transform=training_transform)
    ds._create_pairs()          # repopulate full cohort (overwrites post-split subset)
    return ds


def _train_transform(img_size=224):
    # EXACTLY matches data/datasets.py training augmentation
    return T.Compose([
        T.Resize((img_size + 20, img_size + 20)),
        T.RandomCrop((img_size, img_size)),
        T.RandomHorizontalFlip(0.5),
        T.RandomVerticalFlip(0.2),
        T.RandomRotation(20),
        T.ColorJitter(brightness=0.2, contrast=0.2),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def _eval_transform(img_size=224):
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def get_cv_dataloaders(data_root, folds_json, fold, batch_size=16,
                       num_workers=4, oversample=True, img_size=224):
    """Build train/val DataLoaders for a specific CV fold.

    Returns (loaders_dict, class_weights_tensor).
    loaders_dict has keys "train", "val".
    """
    folds = json.load(open(folds_json))
    assert 0 <= fold < len(folds["folds"]), f"fold {fold} out of range"
    fold_rec = folds["folds"][fold]
    fpairs = folds["pairs"]

    # --- build two dataset views: one with train aug, one with eval transform ---
    ds_train_view = _build_full_cohort_dataset(data_root, _train_transform(img_size))
    ds_eval_view = _build_full_cohort_dataset(data_root, _eval_transform(img_size))
    assert len(ds_train_view.pairs) == len(fpairs), \
        f"cohort size {len(ds_train_view.pairs)} != fold file {len(fpairs)}"

    # --- content-key -> dataset position (ordering-robust) ---
    ds_pos = {_content_key(p): i for i, p in enumerate(ds_train_view.pairs)}

    def resolve(idx_list):
        out = []
        for fi in idx_list:
            k = _content_key(fpairs[fi])
            if k not in ds_pos:
                raise KeyError(f"fold pair {fi} {k} not found in dataset")
            ds_i = ds_pos[k]
            # label consistency guard
            assert ds_train_view.pairs[ds_i]["label"] == fpairs[fi]["label"], \
                f"label mismatch at fold idx {fi}"
            out.append(ds_i)
        return out

    train_pos = resolve(fold_rec["train_idx"])
    val_pos = resolve(fold_rec["val_idx"])

    # --- patient-level disjointness re-assertion ---
    tr_patients = {ds_train_view.pairs[i]["patient_id"] for i in train_pos}
    va_patients = {ds_train_view.pairs[i]["patient_id"] for i in val_pos}
    leak = tr_patients & va_patients
    assert not leak, f"PATIENT LEAKAGE in fold {fold}: {len(leak)} patients in both splits"

    train_subset = Subset(ds_train_view, train_pos)   # train aug
    val_subset = Subset(ds_eval_view, val_pos)         # eval transform

    # --- oversampling sampler (mirrors get_dataloaders) ---
    if oversample:
        labels = [ds_train_view.pairs[i]["label"] for i in train_pos]
        counts = Counter(labels)
        weights = [1.0 / counts[l] for l in labels]
        sampler = WeightedRandomSampler(weights, len(weights) * 2, replacement=True)
        train_loader = DataLoader(train_subset, batch_size=batch_size, sampler=sampler,
                                  num_workers=num_workers, pin_memory=True, drop_last=True)
    else:
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True,
                                  num_workers=num_workers, pin_memory=True, drop_last=True)

    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)

    # --- class weights from this fold's training labels ---
    train_labels = [ds_train_view.pairs[i]["label"] for i in train_pos]
    bc = np.bincount(train_labels, minlength=2)
    cw = len(train_labels) / (2 * bc + 1e-6)
    class_weights = torch.tensor(cw, dtype=torch.float32)

    print(f"[cv] fold {fold}: train={len(train_pos)} ({bc[1]} mal/{bc[0]} ben) "
          f"val={len(val_pos)} | patient-disjoint OK | class_weights={cw.round(3)}")

    return {"train": train_loader, "val": val_loader}, class_weights


if __name__ == "__main__":
    # Smoke test: build fold 0, pull one batch, report shapes
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="/workspace/cbis-ddsm")
    ap.add_argument("--folds_json", default="/workspace/cued_net/cv_folds.json")
    ap.add_argument("--fold", type=int, default=0)
    args = ap.parse_args()

    loaders, cw = get_cv_dataloaders(args.data_root, args.folds_json, args.fold,
                                     batch_size=8, num_workers=2)
    b = next(iter(loaders["train"]))
    print("batch keys:", list(b.keys()))
    print("img_cc:", b["img_cc"].shape, "img_mlo:", b["img_mlo"].shape,
          "label:", b["label"].shape)
    print("label sample:", b["label"].tolist())
    print("SMOKE TEST OK")
