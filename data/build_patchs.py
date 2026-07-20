#!/usr/bin/env python3
"""Extract lesion patches from CBIS-DDSM."""

import argparse, re, sys
from pathlib import Path
from collections import defaultdict
import pandas as pd
from PIL import Image

STUDY_RE = re.compile(r"Mass-(Training|Test)_P_(\d+)_(LEFT|RIGHT)_(CC|MLO)")

def find_col(df, candidates):
    low = {c.lower().strip(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in low:
            return low[cand.lower()]
    return None

def parse_study(s):
    if not isinstance(s, str):
        return None
    m = STUDY_RE.search(s.replace("\\", "/"))
    return (m.group(1), m.group(2), m.group(3), m.group(4)) if m else None

def resolve_jpeg(path_value, kaggle_root, leaf_index):
    val = str(path_value).replace("\\", "/").strip()
    if not val or val == "nan":
        return None
    parts = val.split("/")
    for start in range(len(parts)):
        cand = kaggle_root / Path(*parts[start:])
        if cand.exists():
            return cand
    if len(parts) >= 2:
        key = (parts[-2], parts[-1])
        if key in leaf_index:
            return leaf_index[key]
    return None

def build_leaf_index(kaggle_root):
    idx = {}
    base = kaggle_root / "jpeg"
    it = base.rglob("*.jpg") if base.exists() else kaggle_root.rglob("*.jpg")
    for p in it:
        idx[(p.parent.name, p.name)] = p
    return idx

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/workspace/cbis-ddsm")
    ap.add_argument("--kaggle", default=None)
    ap.add_argument("--dicom_info", default=None)
    ap.add_argument("--series", default="full mammogram images")
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--path_col", default=None)
    ap.add_argument("--study_col", default=None)
    ap.add_argument("--series_col", default=None)
    args = ap.parse_args()

    root = Path(args.root)
    kaggle = Path(args.kaggle) if args.kaggle else root
    out_dir = root / "patchs"; out_dir.mkdir(parents=True, exist_ok=True)

    di_path = Path(args.dicom_info) if args.dicom_info else kaggle / "csv" / "dicom_info.csv"
    if not di_path.exists():
        for alt in [kaggle / "dicom_info.csv", root / "csv" / "dicom_info.csv"]:
            if alt.exists():
                di_path = alt; break
    if not di_path.exists():
        print(f"[!] dicom_info.csv not found at {di_path}."); sys.exit(1)

    print(f"Reading {di_path} ...")
    di = pd.read_csv(di_path)
    path_col = args.path_col or find_col(di, ["image_path","image path","file_path","filepath"])
    study_col = args.study_col or find_col(di, ["PatientID","patient_id","patientid","study"])
    series_col = args.series_col or find_col(di, ["SeriesDescription","series_description"])
    print(f"  columns -> path:{path_col} study:{study_col} series:{series_col}")
    if not all([path_col, study_col, series_col]):
        print(f"[!] auto-detect failed. Available: {list(di.columns)}"); sys.exit(1)

    di = di[di[series_col].astype(str).str.lower().str.strip() == args.series.lower()]
    print(f"  rows with series='{args.series}': {len(di)}")

    print("Indexing jpeg leaves ...")
    leaf_index = build_leaf_index(kaggle)
    print(f"  indexed {len(leaf_index)} jpgs")

    written = unresolved = unparsed = 0
    seen = defaultdict(int)
    for _, row in di.iterrows():
        parsed = parse_study(row[study_col]) or parse_study(str(row[path_col]))
        if not parsed:
            unparsed += 1; continue
        phase, num, lat, view = parsed
        src = resolve_jpeg(row[path_col], kaggle, leaf_index)
        if src is None:
            unresolved += 1; continue
        seen[(phase, num, lat, view)] += 1
        abn = seen[(phase, num, lat, view)]
        out_path = out_dir / f"Mass-{phase}_P_{num}_{lat}_{view}_{abn}_repro.png"
        if out_path.exists():
            written += 1; continue
        try:
            Image.open(src).convert("L").resize((args.size, args.size), Image.BILINEAR).save(out_path)
            written += 1
        except Exception as e:
            print(f"  [!] {src}: {e}"); unresolved += 1

    pat = re.compile(r"Mass-(Training|Test)_P_(\d+)_(LEFT|RIGHT)_(CC|MLO)_(\d+)_")
    groups, phase_of = defaultdict(set), {}
    for p in out_dir.glob("*.png"):
        mm = pat.match(p.name)
        if mm:
            ph, n, l, v, a = mm.groups()
            groups[(n, l, a)].add(v); phase_of[(n, l, a)] = ph
    dev = sum(1 for k,v in groups.items() if {"CC","MLO"} <= v and phase_of[k]=="Training")
    tst = sum(1 for k,v in groups.items() if {"CC","MLO"} <= v and phase_of[k]=="Test")
    print("\n" + "="*56)
    print(f"  wrote {written} PNGs | unresolved {unresolved} | unparsed {unparsed}")
    print("  PAIRING REPORT (both CC+MLO present)")
    print(f"  Dev (train+val) pairs : {dev}   (paper: 493)")
    print(f"  Test pairs            : {tst}   (paper: 149)")
    print(f"  Total                 : {dev+tst}   (paper: 642)")
    print("="*56)

if __name__ == "__main__":
    main()
