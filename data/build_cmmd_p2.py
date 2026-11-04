#!/usr/bin/env python3
"""Build the CMMD cohort for Protocol P2.

Labels are breast-level. Inputs are label-blind whole-breast crops without a
lesion ROI. Patients are assigned once to adaptation training, adaptation
validation and evaluation. Zero-shot models use no CMMD case for any choice.
"""

import argparse
import hashlib
import json
import sys
from multiprocessing import Pool
from pathlib import Path

from PIL import Image
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.build_cmmd_manifest import build as build_pairs
from data.crop_cmmd_breast import breast_bbox, dicom_to_uint8
from training.p2_config import ADAPTATION

SPLIT_SEED = 2026


def patient_split(pairs, train_fraction, val_fraction, seed=SPLIT_SEED):
    """patient id -> adapt_train, adapt_val or eval, stratified by any malignant breast."""
    malignant = {}
    for p in pairs:
        malignant[p["patient_id"]] = max(malignant.get(p["patient_id"], 0), int(p["label"]))
    patients = sorted(malignant)
    strata = [malignant[p] for p in patients]
    adapt, rest = train_test_split(patients, train_size=train_fraction + val_fraction, stratify=strata, random_state=seed)
    train, val = train_test_split(adapt, train_size=train_fraction / (train_fraction + val_fraction),
                                  stratify=[malignant[p] for p in adapt], random_state=seed)
    return {**{p: "adapt_train" for p in train}, **{p: "adapt_val" for p in val}, **{p: "eval" for p in rest}}


def crop_one(job):
    src, dst, size = job
    img = dicom_to_uint8(src)
    y0, y1, x0, x1 = breast_bbox(img, margin_frac=0.04)
    crop = img[y0:y1 + 1, x0:x1 + 1]
    if crop.shape[0] < 32 or crop.shape[1] < 32:
        crop = img
    Image.fromarray(crop).convert("L").resize((size, size), Image.BILINEAR).save(dst)
    return dst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dicom_root", required=True)
    ap.add_argument("--clinical_xlsx", required=True)
    ap.add_argument("--patches", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    raw = str(Path(args.out).with_suffix(".raw.json"))
    build_pairs(args.dicom_root, args.clinical_xlsx, raw)
    records = json.load(open(raw))
    Path(args.patches).mkdir(parents=True, exist_ok=True)
    split = patient_split(records, ADAPTATION["train_fraction"], ADAPTATION["val_fraction"])

    pairs, jobs = [], []
    for r in records:
        key = f"{r['patient_id']}_{r['laterality']}"
        cc, mlo = f"{key}_CC.png", f"{key}_MLO.png"
        jobs += [(r["cc_path"], str(Path(args.patches) / cc), args.size), (r["mlo_path"], str(Path(args.patches) / mlo), args.size)]
        abnormality = (r.get("abnormality") or "").lower()
        pairs.append(dict(key=key, patient_id=r["patient_id"], side=r["laterality"], cc=cc, mlo=mlo, label=int(r["label"]),
                          abnormality=abnormality, contains_mass=abnormality in ("mass", "both"),
                          subtype=r.get("subtype"), age=r.get("age"), split=split[r["patient_id"]]))
    with Pool(args.workers) as pool:
        for i, _ in enumerate(pool.imap_unordered(crop_one, jobs, chunksize=8)):
            if (i + 1) % 1000 == 0:
                print(i + 1, "of", len(jobs), flush=True)

    manifest = dict(pairs=pairs, split_seed=SPLIT_SEED)
    manifest["md5"] = hashlib.md5(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    json.dump(manifest, open(args.out, "w"))
    count = lambda rows: dict(pairs=len(rows), patients=len({p["patient_id"] for p in rows}), malignant=sum(p["label"] for p in rows))
    print(json.dumps({s: count([p for p in pairs if p["split"] == s]) for s in ("adapt_train", "adapt_val", "eval")}
                     | dict(all=count(pairs), mass=count([p for p in pairs if p["contains_mass"]]), md5=manifest["md5"]), indent=1))


if __name__ == "__main__":
    main()
