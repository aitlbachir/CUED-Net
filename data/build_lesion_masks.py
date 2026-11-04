#!/usr/bin/env python3
"""Align CBIS-DDSM lesion masks with the cropped patches.

The ROI masks are full-image binary files. Each crop is located in its full
mammogram by template matching near the mask, and the mask is cut at that
position, then stretched to patch size as the crop was. A match score below
--min_score marks the view as not aligned, and the analysis leaves it out.
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.build_patchs import looks_like_mask, parse_study, resolve_jpeg


def locate(full, crop, centre, scale=4):
    """Return (y, x, score): top-left of `crop` in `full`, searched near `centre`."""
    H, W = crop.shape
    cy, cx = centre
    y0, y1 = max(0, cy - H), min(full.shape[0], cy + H)
    x0, x1 = max(0, cx - W), min(full.shape[1], cx + W)
    area = cv2.resize(full[y0:y1, x0:x1], None, fx=1 / scale, fy=1 / scale, interpolation=cv2.INTER_AREA)
    tmpl = cv2.resize(crop, None, fx=1 / scale, fy=1 / scale, interpolation=cv2.INTER_AREA)
    if tmpl.shape[0] > area.shape[0] or tmpl.shape[1] > area.shape[1] or tmpl.std() == 0:
        return cy - H // 2, cx - W // 2, 0.0
    res = cv2.matchTemplate(area.astype(np.float32), tmpl.astype(np.float32), cv2.TM_CCOEFF_NORMED)
    _, score, _, loc = cv2.minMaxLoc(res)
    return y0 + loc[1] * scale, x0 + loc[0] * scale, float(score)


def cut(mask, y, x, H, W):
    """Window of `mask` at (y, x), zero outside the image."""
    out = np.zeros((H, W), dtype=np.uint8)
    ys, xs = max(0, y), max(0, x)
    ye, xe = min(mask.shape[0], y + H), min(mask.shape[1], x + W)
    if ye > ys and xe > xs:
        out[ys - y:ye - y, xs - x:xe - x] = mask[ys:ye, xs:xe]
    return out


def align(full, mask, crop):
    if mask.shape != full.shape:
        mask = cv2.resize(mask, (full.shape[1], full.shape[0]), interpolation=cv2.INTER_NEAREST)
    ys, xs = np.nonzero(mask > 127)
    if len(ys) == 0:
        return None, 0.0
    centre = (int((ys.min() + ys.max()) / 2), int((xs.min() + xs.max()) / 2))
    y, x, score = locate(full, crop, centre)
    return cut((mask > 127).astype(np.uint8) * 255, y, x, *crop.shape), score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kaggle", required=True)
    ap.add_argument("--patch_report", required=True, help="report csv from build_patchs.py")
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--min_score", type=float, default=0.8)
    args = ap.parse_args()

    root, out = Path(args.kaggle), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    di = pd.read_csv(next(root.rglob("dicom_info.csv")))
    series = di["SeriesDescription"].astype(str).str.lower().str.strip()
    leaf_index = {(p.parent.name, p.name): p for p in root.rglob("*.jpg")}
    masks, fulls = {}, {}
    for _, row in di[series == "roi mask images"].iterrows():
        masks.setdefault(str(row["PatientID"]), []).append(row["image_path"])
    for _, row in di[series == "full mammogram images"].iterrows():
        parsed = parse_study(row["PatientID"])
        if parsed:
            fulls[parsed[:4]] = row["image_path"]

    rows = []
    for rec in pd.read_csv(args.patch_report).itertuples():
        # Development cases only. The held-out split is not read.
        if rec.duplicate or "Mass-Training" not in rec.file:
            continue
        crop = np.asarray(Image.open(rec.source).convert("L"))
        parsed = parse_study(rec.study)
        status, score = "no_mask", 0.0
        full_value = fulls.get(parsed[:4])
        full_path = resolve_jpeg(full_value, root, leaf_index) if full_value else None
        best = None
        for candidate in masks.get(rec.study, []):
            path = resolve_jpeg(candidate, root, leaf_index)
            if path is None or full_path is None:
                continue
            m = np.asarray(Image.open(path).convert("L"))
            if not looks_like_mask(m):
                continue
            aligned, s = align(np.asarray(Image.open(full_path).convert("L")), m, crop)
            if aligned is not None and s > score:
                best, score = aligned, s
        if best is not None:
            status = "aligned" if score >= args.min_score else "low_score"
            Image.fromarray(best).resize((args.size, args.size), Image.NEAREST).save(out / rec.file)
        rows.append(dict(file=rec.file, status=status, score=score, mask_fraction=float((best > 0).mean()) if best is not None else np.nan))
    table = pd.DataFrame(rows)
    table.to_csv(out / "masks_report.csv", index=False)
    print(table.status.value_counts().to_string())
    print(table.groupby("status")[["score", "mask_fraction"]].mean().round(3).to_string())


if __name__ == "__main__":
    main()
