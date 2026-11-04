#!/usr/bin/env python3
"""Build the P2 lesion-level manifest.

Pairs come from the CBIS-DDSM mass case tables, keyed on patient, side and
abnormality id. Fold membership comes from the frozen R1 manifest, by patient.
A pair is kept only if both patches exist and CC and MLO carry the same label.
"""

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd

NAME = re.compile(r"Mass-(Training|Test)_P_(\d+)_(LEFT|RIGHT)_(CC|MLO)_(\d+)_repro\.png")


def read_cases(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.replace(" ", "_")
    return df


def label_of(pathology):
    p = str(pathology).upper()
    return 1 if "MALIGNANT" in p else 0 if "BENIGN" in p else -1


def pairs_for(patch_dir, cases, phase, audit):
    images = {}
    for f in Path(patch_dir).glob(f"Mass-{phase}_*.png"):
        m = NAME.match(f.name)
        images[(f"P_{m.group(2)}", m.group(3), m.group(4), int(m.group(5)))] = f.name
    labels = {(r.patient_id, r.left_or_right_breast, r.image_view, int(r.abnormality_id)): label_of(r.pathology)
              for r in cases.itertuples()}
    keys = sorted({k[:2] + (k[3],) for k in images})
    out = []
    for patient, side, abn in keys:
        cc, mlo = images.get((patient, side, "CC", abn)), images.get((patient, side, "MLO", abn))
        if not (cc and mlo):
            audit["missing_view"] += 1
            continue
        lc, lm = labels.get((patient, side, "CC", abn), -1), labels.get((patient, side, "MLO", abn), -1)
        if lc < 0 or lm < 0:
            audit["no_label"] += 1
            continue
        if lc != lm:
            audit["label_conflict"] += 1
            continue
        out.append(dict(key=f"{patient}_{side}_{abn}", patient_id=patient, side=side, abnormality=abn,
                        cc=cc, mlo=mlo, label=lc))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patches", required=True)
    ap.add_argument("--train_csv", required=True)
    ap.add_argument("--test_csv", required=True)
    ap.add_argument("--frozen_folds", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    audit = Counter()
    pairs = pairs_for(args.patches, read_cases(args.train_csv), "Training", audit)
    test_pairs = pairs_for(args.patches, read_cases(args.test_csv), "Test", audit)

    frozen = json.load(open(args.frozen_folds))
    fold_of = {}
    for rec in frozen["folds"]:
        for i in rec["val_idx"]:
            fold_of[frozen["pairs"][i]["patient_id"]] = rec["fold"]
    missing = sorted({p["patient_id"] for p in pairs} - set(fold_of))
    audit["patients_outside_frozen_folds"] = len(missing)
    pairs = [p for p in pairs if p["patient_id"] in fold_of]

    folds = []
    for k in range(len(frozen["folds"])):
        folds.append(dict(fold=k, train_idx=[i for i, p in enumerate(pairs) if fold_of[p["patient_id"]] != k],
                          val_idx=[i for i, p in enumerate(pairs) if fold_of[p["patient_id"]] == k]))
    overlap = {p["patient_id"] for p in pairs} & {p["patient_id"] for p in test_pairs}
    manifest = dict(pairs=pairs, test_pairs=test_pairs, folds=folds, audit=dict(audit), test_overlap=sorted(overlap))
    body = json.dumps(manifest, sort_keys=True).encode()
    manifest["md5"] = hashlib.md5(body).hexdigest()
    json.dump(manifest, open(args.out, "w"))
    print(json.dumps(dict(n_pairs=len(pairs), n_patients=len({p["patient_id"] for p in pairs}),
                          malignant=sum(p["label"] for p in pairs), n_test=len(test_pairs),
                          n_test_patients=len({p["patient_id"] for p in test_pairs}),
                          malignant_test=sum(p["label"] for p in test_pairs), overlap=len(overlap),
                          audit=audit, md5=manifest["md5"]), indent=1))


if __name__ == "__main__":
    main()
