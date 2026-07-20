#!/usr/bin/env python3
"""
build_cmmd_manifest_nbia.py
===========================
CC+MLO lesion-pair manifest for the FULL CMMD external-validation set, as
downloaded from TCIA/NBIA (flat, UID-named series directories).

Differs from the Kaggle-mirror version ONLY in data discovery:
  - iterates  cmmd_full/<SeriesUID>/*.dcm   (not  CMMD/D1-XXXX/...)
  - patient_id is read from DICOM (0010,0020) PatientID  (= D1-/D2- ID),
    NOT from the directory name (which is a SeriesInstanceUID)
  - groups by (patient_id, laterality) across ALL files in ALL series dirs,
    so a bilateral patient split across series still assembles correctly

Identical to the validated version:
  view   : (0054,0220) ViewCodeSequence CodeValue  399162004=CC / 399368009=MLO
  lat    : (0020,0062) ImageLaterality  R/L
  label  : clinical xlsx join on (ID1, LeftRight)  Malignant->1 / Benign->0
  pair   : emitted only when both CC and MLO exist for that breast
  report : full drop accounting + reconciliation assertion + hard invariants

CMMD is EXTERNAL VALIDATION ONLY — never enters CV/training splits.
"""

import os
import glob
import json
import argparse
from collections import defaultdict

import pandas as pd
import pydicom

CC_CODES  = {"399162004"}
MLO_CODES = {"399368009"}
LABEL_MAP = {"malignant": 1, "benign": 0}


def read_header_fields(dcm_path):
    """Return (patient_id, laterality, view) or (...,None,...) on any miss."""
    d = pydicom.dcmread(dcm_path, stop_before_pixels=True)

    pid = str(d.get("PatientID", "")).strip()
    if not pid:
        return None, None, None

    lat = str(d.get("ImageLaterality", "") or d.get("Laterality", "")).strip().upper()
    if lat not in ("R", "L"):
        return pid, None, None

    view = None
    vcs = d.get("ViewCodeSequence", None)
    if vcs is not None and len(vcs) > 0:
        cv = str(vcs[0].get("CodeValue", "")).strip()
        if cv in CC_CODES:
            view = "CC"
        elif cv in MLO_CODES:
            view = "MLO"
    return pid, lat, view


def build(dicom_root, clinical_xlsx, out_path):
    # ---- 1. clinical labels, keyed (ID1, LeftRight) -----------------------
    clin = pd.read_excel(clinical_xlsx)
    clin.columns = [c.strip() for c in clin.columns]
    clin["LeftRight"] = clin["LeftRight"].astype(str).str.strip().str.upper()
    clin["ID1"] = clin["ID1"].astype(str).str.strip()
    label_lookup, meta_lookup = {}, {}
    for _, r in clin.iterrows():
        cls = str(r["classification"]).strip().lower()
        if cls not in LABEL_MAP:
            continue
        key = (r["ID1"], r["LeftRight"])
        label_lookup[key] = LABEL_MAP[cls]
        meta_lookup[key] = {
            "abnormality": (None if pd.isna(r.get("abnormality")) else str(r.get("abnormality"))),
            "subtype": (None if pd.isna(r.get("subtype")) else str(r.get("subtype"))),
            "age": (None if pd.isna(r.get("Age")) else int(r.get("Age"))),
        }

    # ---- 2. scan ALL dicoms across ALL series dirs, group by (pid, lat) ---
    # views[(pid,lat)] = {"CC":path, "MLO":path}
    views = defaultdict(dict)
    drops = defaultdict(int)
    all_dcm = glob.glob(os.path.join(dicom_root, "*", "*.dcm"))
    all_dcm += glob.glob(os.path.join(dicom_root, "*", "**", "*.dcm"), recursive=True)
    all_dcm = sorted(set(all_dcm))
    for f in all_dcm:
        pid, lat, view = read_header_fields(f)
        if pid is None:
            drops["no_patient_id"] += 1
            continue
        if lat is None:
            drops["no_laterality"] += 1
            continue
        if view is None:
            drops["no_view_code"] += 1
            continue
        slot = views[(pid, lat)]
        if view in slot:
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
    print("CMMD (full TCIA/NBIA) manifest build report")
    print("=" * 64)
    print(f"dicom files scanned             : {len(all_dcm)}")
    print(f"clinical breast-rows (labelled) : {n_clinical_rows}")
    print(f"emitted CC+MLO pairs            : {n_pairs}")
    print(f"  malignant (1)                 : {n_pos}")
    print(f"  benign    (0)                 : {n_neg}")
    if n_pairs:
        print(f"  malignant ratio               : {n_pos / n_pairs:.4f}")
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

    for r in records:
        assert r["label"] in (0, 1)
        assert os.path.exists(r["cc_path"]) and os.path.exists(r["mlo_path"])
        assert r["cc_path"] != r["mlo_path"]
    print("invariants: 2 distinct existing files per pair, label in {0,1}  OK")

    # coverage note vs full cohort
    covered = {(r["patient_id"], r["laterality"]) for r in records}
    clinical_keys = set(label_lookup.keys())
    missing = sorted(clinical_keys - covered)
    print(f"clinical breasts NOT emitted    : {len(missing)}")
    if missing[:10]:
        print(f"  first 10                      : {missing[:10]}")

    with open(out_path, "w") as fh:
        json.dump(records, fh, indent=2)
    print(f"\nwrote {n_pairs} records -> {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dicom_root", default="/workspace/cmmd_full")
    ap.add_argument("--clinical_xlsx",
                    default="/workspace/cmmd_external/CMMD_clinicaldata_revision.xlsx")
    ap.add_argument("--out", default="/workspace/cued_net/cmmd_pairs_full.json")
    args = ap.parse_args()
    build(args.dicom_root, args.clinical_xlsx, args.out)