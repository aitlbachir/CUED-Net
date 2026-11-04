#!/usr/bin/env python3
"""Extract CBIS-DDSM mass patches from the Kaggle JPEG release.

Series 'cropped images' holds the pre-cropped lesion ROIs used by CUED-Net.

--id_source study    abnormality id read from the study name, for example
                     Mass-Training_P_00001_LEFT_CC_1. Use this for R2.
--id_source order    id assigned by row order. This reproduces the R1 patch set
                     and exists only for the provenance check.
"""

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

STUDY_RE = re.compile(r"Mass-(Training|Test)_P_(\d+)_(LEFT|RIGHT)_(CC|MLO)(?:_(\d+))?")


def parse_study(value):
    m = STUDY_RE.search(str(value).replace("\\", "/"))
    return m.groups() if m else None


def resolve_jpeg(value, root, leaf_index):
    parts = str(value).replace("\\", "/").strip().split("/")
    for start in range(len(parts)):
        cand = root / Path(*parts[start:])
        if cand.exists():
            return cand
    return leaf_index.get(tuple(parts[-2:]))


def looks_like_mask(arr):
    """A binary image labelled as a crop."""
    values = np.unique(arr)
    return len(values) <= 4 or np.mean((arr == 0) | (arr == 255)) > 0.95


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kaggle", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--series", default="cropped images")
    ap.add_argument("--id_source", choices=["study", "order"], default="study")
    ap.add_argument("--size", type=int, default=224)
    args = ap.parse_args()

    root, out = Path(args.kaggle), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    di = pd.read_csv(next(root.rglob("dicom_info.csv")))
    di = di[di["SeriesDescription"].astype(str).str.lower().str.strip() == args.series]
    leaf_index = {(p.parent.name, p.name): p for p in root.rglob("*.jpg")}

    seen, written, report = defaultdict(int), Counter(), []
    for _, row in di.iterrows():
        parsed = parse_study(row["PatientID"]) or parse_study(row["image_path"])
        if not parsed:
            written["not_mass"] += 1
            continue
        phase, num, lat, view, study_id = parsed
        src = resolve_jpeg(row["image_path"], root, leaf_index)
        if src is None:
            written["unresolved"] += 1
            continue
        key = (phase, num, lat, view)
        seen[key] += 1
        abn = seen[key] if args.id_source == "order" else (study_id or seen[key])
        path = out / f"Mass-{phase}_P_{num}_{lat}_{view}_{abn}_repro.png"
        im = Image.open(src).convert("L")
        arr = np.asarray(im)
        mask_like = looks_like_mask(arr)
        duplicate = path.exists()
        im.resize((args.size, args.size), Image.BILINEAR).save(path)
        written["duplicate" if duplicate else "written"] += 1
        written["mask_like"] += int(mask_like)
        report.append(dict(file=path.name, source=str(src), study=str(row["PatientID"]), study_id=study_id,
                           order_id=seen[key], height=arr.shape[0], width=arr.shape[1], mean=float(arr.mean()),
                           mask_like=bool(mask_like), duplicate=duplicate))
    pd.DataFrame(report).to_csv(out.parent / f"{out.name}_report.csv", index=False)
    print(json.dumps(written, indent=1))


if __name__ == "__main__":
    main()
