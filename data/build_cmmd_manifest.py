#!/usr/bin/env python3
"""
build_cmmd_manifest.py
======================
Construct a leakage-free, header-verified CC+MLO lesion-pair manifest for the
CMMD external-validation set, in the same pair schema consumed by
CBISDDSMDataset.

Sample unit  : (patient_id, laterality) breast-pair
Pair rule    : created only when BOTH a CC and an MLO DICOM exist for that breast
View source  : DICOM (0054,0220) ViewCodeSequence -> CodeValue
                 399162004 = CC, 399368009 = MLO   (SNOMED-CT)
Laterality   : DICOM (0020,0062) ImageLaterality (R/L)
Label        : clinical xlsx, joined on (ID1, LeftRight)
                 Malignant -> 1, Benign -> 0
Group key    : patient_id   (= ID1; for CV/eval grouping, no patient split here)

Output: cmmd_pairs.json  — list of records:
    {
      "patient_id": "D2-0001",
      "laterality": "L",
      "label": 1,
      "cc_path":  "<abs path to CC dicom>",
      "mlo_path": "<abs path to MLO dicom>",
      "abnormality": "calcification",
      "subtype": "...",            # may be null
      "age": 44
    }

CMMD is EXTERNAL VALIDATION ONLY. It must never enter CV-fold construction
or any training split. This script does not split; it emits the full manifest.
"""

import os
import glob
import json
import argparse
from collections import defaultdict

import pandas as pd
import pydicom

# SNOMED-CT view codes used by CMMD
CC_CODES  = {"399162004"}
MLO_CODES = {"399368009"}

LABEL_MAP = {"malignant": 1, "benign": 0}


def read_view_laterality(dcm_path):
    """Return (laterality, view) for one DICOM, or (None, None) if unrecoverable.
    view in {'CC','MLO'}; laterality in {'R','L'}."""
    d = pydicom.dcmread(dcm_path, stop_before_pixels=True)

    lat = str(d.get("ImageLaterality", "") or d.get("Laterality", "")).strip().upper()
    if lat not in ("R", "L"):
        return None, None

    view = None
    vcs = d.get("ViewCodeSequence", None)
    if vcs is not None and len(vcs) > 0:
        cv = str(vcs[0].get("CodeValue", "")).strip()
        if cv in CC_CODES:
            view = "CC"
        elif cv in MLO_CODES:
            view = "MLO"
    return lat, view


def build(dicom_root, clinical_xlsx, out_path):
    # ---- 1. clinical labels, keyed (ID1, LeftRight) -----------------------
    clin = pd.read_excel(clinical_xlsx)
    clin.columns = [c.strip() for c in clin.columns]
    # one row per (patient, breast)
    clin["LeftRight"] = clin["LeftRight"].astype(str).str.strip().str.upper()
    clin["ID1"] = clin["ID1"].astype(str).str.strip()
    label_lookup = {}
    meta_lookup = {}
    for _, r in clin.iterrows():
        key = (r["ID1"], r["LeftRight"])
        cls = str(r["classification"]).strip().lower()
        if cls not in LABEL_MAP:
            continue
        label_lookup[key] = LABEL_MAP[cls]
        meta_lookup[key] = {
            "abnormality": (None if pd.isna(r.get("abnormality"))
                            else str(r.get("abnormality"))),
            "subtype": (None if pd.isna(r.get("subtype"))
                        else str(r.get("subtype"))),
            "age": (None if pd.isna(r.get("Age")) else int(r.get("Age"))),
        }

    # ---- 2. scan DICOMs, group by (patient, laterality) -------------------
    # views[(pid,lat)] = {"CC":path, "MLO":path}
    views = defaultdict(dict)
    drops = defaultdict(int)
    patient_dirs = sorted(
        d for d in glob.glob(os.path.join(dicom_root, "D*"))
        if os.path.isdir(d)
    )
    for pdir in patient_dirs:
        pid = os.path.basename(pdir)
        files = sorted(glob.glob(os.path.join(pdir, "**", "*.dcm"),
                                 recursive=True))
        for f in files:
            lat, view = read_view_laterality(f)
            if lat is None:
                drops["no_laterality"] += 1
                continue
            if view is None:
                drops["no_view_code"] += 1
                continue
            slot = views[(pid, lat)]
            if view in slot:
                # duplicate same-view image for this breast: keep first, note it
                drops[f"duplicate_{view}"] += 1
                continue
            slot[view] = os.path.abspath(f)

    # ---- 3. emit pairs: require BOTH views AND a matched label ------------
    records = []
    n_no_pair, n_no_label = 0, 0
    for (pid, lat), slot in views.items():
        if "CC" not in slot or "MLO" not in slot:
            n_no_pair += 1
            continue
        key = (pid, lat)
        if key not in label_lookup:
            n_no_label += 1
            continue
        meta = meta_lookup[key]
        records.append({
            "patient_id": pid,
            "laterality": lat,
            "label": label_lookup[key],
            "cc_path": slot["CC"],
            "mlo_path": slot["MLO"],
            "abnormality": meta["abnormality"],
            "subtype": meta["subtype"],
            "age": meta["age"],
        })

    records.sort(key=lambda r: (r["patient_id"], r["laterality"]))

    # ---- 4. validation + reconciliation ----------------------------------
    n_pairs = len(records)
    n_pos = sum(r["label"] for r in records)
    n_neg = n_pairs - n_pos
    n_patients = len({r["patient_id"] for r in records})
    n_clinical_rows = len(label_lookup)

    print("=" * 64)
    print("CMMD manifest build report")
    print("=" * 64)
    print(f"patient folders scanned         : {len(patient_dirs)}")
    print(f"clinical breast-rows (labelled) : {n_clinical_rows}")
    print(f"emitted CC+MLO pairs            : {n_pairs}")
    print(f"  malignant (1)                 : {n_pos}")
    print(f"  benign    (0)                 : {n_neg}")
    print(f"  malignant ratio               : {n_pos / n_pairs:.4f}" if n_pairs else "  n/a")
    print(f"distinct patients in manifest   : {n_patients}")
    print("-" * 64)
    print("dropped (with reason):")
    for k, v in sorted(drops.items()):
        print(f"  {k:24s}: {v}")
    print(f"  breast w/o complete CC+MLO    : {n_no_pair}")
    print(f"  breast w/o clinical label     : {n_no_label}")
    print("-" * 64)
    reconciled = n_pairs + n_no_pair + n_no_label
    print(f"reconciliation: pairs + no_pair + no_label = {reconciled}")
    print(f"               vs (patient,lat) groups seen = {len(views)}")
    assert reconciled == len(views), "RECONCILIATION FAILED — investigate"

    # hard invariants
    for r in records:
        assert r["label"] in (0, 1)
        assert os.path.exists(r["cc_path"])
        assert os.path.exists(r["mlo_path"])
        assert r["cc_path"] != r["mlo_path"]
    print("invariants: all pairs have 2 distinct existing files, label in {0,1}  OK")

    with open(out_path, "w") as fh:
        json.dump(records, fh, indent=2)
    print(f"\nwrote {n_pairs} records -> {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dicom_root",
        default="/workspace/cmmd_external/cmmd_data/"
                "TheChineseMammographyDatabase/CMMD",
    )
    ap.add_argument(
        "--clinical_xlsx",
        default="/workspace/cmmd_external/CMMD_clinicaldata_revision.xlsx",
    )
    ap.add_argument(
        "--out",
        default="/workspace/cued_net/cmmd_pairs.json",
    )
    args = ap.parse_args()
    build(args.dicom_root, args.clinical_xlsx, args.out)