"""
build_cv_folds.py
5-fold StratifiedGroupKFold over CUED-Net training pairs, grouped by patient.

Recovered semantics (from data/datasets.py):
  sample  = CC+MLO lesion-pair (patient_id, laterality, abnormality_id)
  label   = MALIGNANT->1, BENIGN->0 (BWC->0), uncertain dropped
  group   = patient_id (leakage guarantee)
  stratify= per-patient label = max over that patient's pair labels
Pairs are read from CBISDDSMDataset.pairs so they are identical to the run.
"""
import json, argparse
from pathlib import Path
from collections import defaultdict
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from data.datasets import CBISDDSMDataset


def load_pairs(root_dir):
    import pandas as pd
    # Build the full pair list from patches (image_info covers ALL patches on disk).
    ds = CBISDDSMDataset(root_dir=root_dir, split='train')
    ds._create_pairs()                       # repopulate full pair list
    all_pairs = list(ds.pairs)

    # Authoritative train scope: patients present in the TRAIN CSV only.
    csv_path = Path(root_dir) / 'mass_case_description_train_set.csv'
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip().str.replace(' ', '_')
    train_patients = set('P_' + df['patient_id'].astype(str).str.extract(r'P_(\d+)')[0])

    # Keep only pairs whose patient is in the train CSV -> no test-set leakage.
    pairs = [p for p in all_pairs if p['patient_id'] in train_patients]
    if not pairs:
        raise RuntimeError("No train-scoped pairs; check CSV/patch patient-id formats.")
    return pairs


def patient_level_labels(pairs):
    by_patient = defaultdict(list)
    for p in pairs:
        by_patient[p['patient_id']].append(int(p['label']))
    return {pid: int(max(lbls)) for pid, lbls in by_patient.items()}


def build_folds(pairs, n_splits=5, seed=42):
    n = len(pairs)
    groups = np.array([p['patient_id'] for p in pairs])
    pat_lbl = patient_level_labels(pairs)
    y = np.array([pat_lbl[p['patient_id']] for p in pairs])
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = []
    for k, (tr, va) in enumerate(sgkf.split(np.zeros(n), y, groups)):
        folds.append({'fold': k, 'train_idx': tr.tolist(), 'val_idx': va.tolist(),
                      'val_patients': sorted(set(groups[va].tolist()))})
    return folds


def summarize(pairs, folds):
    labels = np.array([int(p['label']) for p in pairs])
    groups = np.array([p['patient_id'] for p in pairs])
    print(f"Population: {len(pairs)} pairs | {len(set(groups))} patients | "
          f"global malignant ratio (pair-level) = {labels.mean():.4f}")
    print("-" * 64)
    for f in folds:
        va = np.array(f['val_idx']); vlab = labels[va]
        print(f"fold {f['fold']}: train_pairs={len(f['train_idx']):4d}  "
              f"val_pairs={len(va):4d}  val_patients={len(f['val_patients']):4d}  "
              f"val_malignant_ratio={vlab.mean():.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root_dir', default='/workspace/cbis-ddsm')
    ap.add_argument('--n_splits', type=int, default=5)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out', default='/workspace/cued_net/cv_folds.json')
    args = ap.parse_args()
    pairs = load_pairs(args.root_dir)
    folds = build_folds(pairs, n_splits=args.n_splits, seed=args.seed)
    summarize(pairs, folds)
    manifest = {'root_dir': args.root_dir, 'n_splits': args.n_splits, 'seed': args.seed,
                'pairs': [{'patient_id': p['patient_id'], 'cc_path': str(p['cc_path']),
                           'mlo_path': str(p['mlo_path']), 'label': int(p['label'])}
                          for p in pairs],
                'folds': folds}
    with open(args.out, 'w') as fh:
        json.dump(manifest, fh, indent=2)
    print("-" * 64); print(f"Wrote {args.out}")


if __name__ == '__main__':
    main()
