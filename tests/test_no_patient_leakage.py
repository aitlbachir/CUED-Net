"""
test_no_patient_leakage.py
Verifies the 5-fold CV: HARD patient-disjointness (H1-H4), SOFT balance (S1).
"""
import json, argparse
from collections import defaultdict
import numpy as np
from build_cv_folds import load_pairs, build_folds

TOL = 0.10


def _checks(pairs, folds):
    groups = np.array([p['patient_id'] for p in pairs])
    labels = np.array([int(p['label']) for p in pairs])
    all_patients = set(groups.tolist())
    global_ratio = labels.mean()
    val_sets = [set(f['val_patients']) for f in folds]

    for i in range(len(folds)):
        for j in range(i + 1, len(folds)):
            inter = val_sets[i] & val_sets[j]
            assert not inter, f"[H1 FAIL] val folds {i}&{j} share {sorted(inter)[:5]}"
    print("[H1 PASS] validation patient sets are pairwise disjoint")

    for f in folds:
        tr_pat = set(groups[np.array(f['train_idx'])].tolist())
        va_pat = set(f['val_patients'])
        inter = tr_pat & va_pat
        assert not inter, f"[H2 FAIL] fold {f['fold']} train&val share {sorted(inter)[:5]}"
    print("[H2 PASS] train/val patient sets disjoint within every fold")

    union = set().union(*val_sets)
    assert union == all_patients, "[H3 FAIL] val union != all patients"
    assert sum(len(s) for s in val_sets) == len(all_patients), "[H3 FAIL] patient validated >once"
    print(f"[H3 PASS] {len(all_patients)} patients each validated exactly once")

    pbp = defaultdict(list)
    for idx, p in enumerate(pairs):
        pbp[p['patient_id']].append(idx)
    for f in folds:
        va_idx, tr_idx = set(f['val_idx']), set(f['train_idx'])
        for pid in f['val_patients']:
            owned = set(pbp[pid])
            assert owned <= va_idx, f"[H4 FAIL] fold {f['fold']}: pairs of {pid} not all in val"
            assert not (owned & tr_idx), f"[H4 FAIL] fold {f['fold']}: pairs of {pid} in train"
    print("[H4 PASS] all pairs of a val patient are in val, none in train")

    print("-" * 64)
    print(f"global malignant ratio (pair-level) = {global_ratio:.4f}")
    soft_ok = True
    for f in folds:
        va = np.array(f['val_idx']); r = labels[va].mean()
        flag = "" if abs(r - global_ratio) <= TOL else "  <-- exceeds tol"
        if flag: soft_ok = False
        print(f"  fold {f['fold']}: val_malignant_ratio={r:.4f} (|delta|={abs(r-global_ratio):.4f}){flag}")
    print(f"[S1 {'PASS' if soft_ok else 'WARN'}] tolerance +/-{TOL:.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root_dir', default='/workspace/cbis-ddsm')
    ap.add_argument('--n_splits', type=int, default=5)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--manifest', default=None)
    args = ap.parse_args()
    if args.manifest:
        with open(args.manifest) as fh:
            m = json.load(fh)
        pairs, folds = m['pairs'], m['folds']
        print(f"Verifying manifest: {args.manifest} ({len(pairs)} pairs, {len(folds)} folds)")
    else:
        pairs = load_pairs(args.root_dir)
        folds = build_folds(pairs, n_splits=args.n_splits, seed=args.seed)
        print(f"Built in-memory: {len(pairs)} pairs, {len(folds)} folds")
    print("=" * 64); _checks(pairs, folds); print("=" * 64)
    print("ALL HARD ASSERTIONS PASSED — no patient crosses folds.")


if __name__ == '__main__':
    main()
