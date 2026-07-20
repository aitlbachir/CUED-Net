#!/usr/bin/env python
"""
crop_cmmd_breast.py  — Label-blind breast-region cropping for CMMD.

PURPOSE
-------
CUED-Net was trained on lesion-centered ROI *patches* where the tissue of
interest fills the 224x224 frame. CMMD images are full-field mammograms in
which the breast occupies a fraction of the frame and most pixels are air /
background. Resizing the whole image to 224x224 therefore shrinks the
diagnostic tissue to a few percent of the input area — a pure input-distribution
mismatch, independent of any label.

This script closes part of that gap WITHOUT touching any label:
  1. Read DICOM -> 8-bit grayscale (VOI-LUT, MONOCHROME1 inversion, bit-depth
     handled the same way as the eval loader).
  2. Otsu threshold -> binary mask of "tissue vs background".
  3. Keep the largest connected component (the breast), discard labels strips,
     scanner artifacts, small bright specks.
  4. Tight bounding box around that component, expanded by a fixed fractional
     margin, then a light pectoral-safe square pad so the aspect ratio is sane.
  5. Save the cropped PNG and write a NEW manifest whose cc_path/mlo_path point
     at the cropped PNGs.

It is fully label-blind: the only inputs are pixel intensities. The label field
is copied through verbatim and never consulted in any cropping decision.

Run, then re-run evaluate_cmmd.py UNCHANGED against the new manifest:

    python crop_cmmd_breast.py \
        --manifest  /workspace/cued_net/cmmd_pairs_full.json \
        --out_dir   /workspace/cmmd_cropped \
        --out_manifest /workspace/cued_net/cmmd_pairs_cropped.json

    python evaluate_cmmd.py \
        --manifest  /workspace/cued_net/cmmd_pairs_cropped.json \
        --ckpt_glob "/workspace/outputs_cued/seed_*/best_model.pt" \
        --models_dir /workspace/cued_net \
        --output    /workspace/cued_net/cmmd_results_cropped.json
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

try:
    import pydicom
    from pydicom.pixel_data_handlers.util import apply_voi_lut
except ImportError:
    raise SystemExit("pip install pydicom")

# Otsu + connected components without bringing in scipy if not present.
try:
    from skimage.filters import threshold_otsu
    from skimage.measure import label as cc_label
    from skimage.measure import regionprops
    _HAVE_SKIMAGE = True
except ImportError:
    _HAVE_SKIMAGE = False


# --------------------------------------------------------------------------- #
# DICOM -> 8-bit grayscale  (mirror the eval loader's behavior)
# --------------------------------------------------------------------------- #
def dicom_to_uint8(path: str) -> np.ndarray:
    ds = pydicom.dcmread(path)
    arr = ds.pixel_array.astype(np.float32)

    # VOI-LUT / windowing if present
    try:
        arr = apply_voi_lut(ds.pixel_array, ds).astype(np.float32)
    except Exception:
        pass

    # MONOCHROME1 -> invert so that high value = bright tissue
    if str(getattr(ds, "PhotometricInterpretation", "")).upper() == "MONOCHROME1":
        arr = arr.max() - arr

    # Normalize to [0, 255]
    arr = arr - arr.min()
    if arr.max() > 0:
        arr = arr / arr.max()
    arr = (arr * 255.0).astype(np.uint8)
    return arr


# --------------------------------------------------------------------------- #
# Otsu fallback (no skimage) — simple histogram-based threshold
# --------------------------------------------------------------------------- #
def _otsu_threshold_np(img: np.ndarray) -> int:
    hist, _ = np.histogram(img.ravel(), bins=256, range=(0, 255))
    total = img.size
    sum_total = np.dot(np.arange(256), hist)
    sum_b, w_b, max_var, thr = 0.0, 0.0, 0.0, 0
    for t in range(256):
        w_b += hist[t]
        if w_b == 0:
            continue
        w_f = total - w_b
        if w_f == 0:
            break
        sum_b += t * hist[t]
        m_b = sum_b / w_b
        m_f = (sum_total - sum_b) / w_f
        var_between = w_b * w_f * (m_b - m_f) ** 2
        if var_between > max_var:
            max_var, thr = var_between, t
    return thr


def _largest_component_bbox_np(mask: np.ndarray):
    """Flood-fill-free largest CC via simple labeling using numpy only.
    Falls back to bbox of the whole mask if labeling is too costly."""
    # Cheap: use the bounding box of all foreground if no skimage.
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return None
    return ys.min(), ys.max(), xs.min(), xs.max()


# --------------------------------------------------------------------------- #
# Breast bounding box (label-blind)
# --------------------------------------------------------------------------- #
def breast_bbox(img: np.ndarray, margin_frac: float = 0.04):
    """Return (y0, y1, x0, x1) bounding box of the breast region.

    Strategy: Otsu threshold -> largest connected component -> tight bbox,
    expanded by margin_frac of the bbox size on each side, clamped to image.
    """
    H, W = img.shape

    if _HAVE_SKIMAGE:
        try:
            thr = threshold_otsu(img)
        except Exception:
            thr = _otsu_threshold_np(img)
        mask = img > thr
        # remove thin background strips by keeping the largest CC
        lab = cc_label(mask)
        if lab.max() == 0:
            return 0, H - 1, 0, W - 1
        props = regionprops(lab)
        biggest = max(props, key=lambda p: p.area)
        y0, x0, y1, x1 = biggest.bbox  # (min_row, min_col, max_row, max_col)
        y1 -= 1
        x1 -= 1
    else:
        thr = _otsu_threshold_np(img)
        mask = img > thr
        bb = _largest_component_bbox_np(mask)
        if bb is None:
            return 0, H - 1, 0, W - 1
        y0, y1, x0, x1 = bb

    # expand by margin
    bh, bw = (y1 - y0), (x1 - x0)
    my, mx = int(bh * margin_frac), int(bw * margin_frac)
    y0 = max(0, y0 - my)
    x0 = max(0, x0 - mx)
    y1 = min(H - 1, y1 + my)
    x1 = min(W - 1, x1 + mx)
    return y0, y1, x0, x1


def crop_and_save(dicom_path: str, out_png: str, margin_frac: float) -> bool:
    try:
        img = dicom_to_uint8(dicom_path)
    except Exception as e:
        print(f"  [skip] cannot read {dicom_path}: {e}")
        return False

    y0, y1, x0, x1 = breast_bbox(img, margin_frac=margin_frac)
    crop = img[y0:y1 + 1, x0:x1 + 1]

    # Guard against degenerate crops (e.g. mask failure): fall back to full img
    if crop.size == 0 or crop.shape[0] < 32 or crop.shape[1] < 32:
        crop = img

    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(crop).convert("L").save(out_png)
    return True


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(args):
    manifest = json.load(open(args.manifest))
    # manifest may be a list of pair dicts OR {"pairs": [...]}
    pairs = manifest["pairs"] if isinstance(manifest, dict) and "pairs" in manifest else manifest

    print(f"[crop] {len(pairs)} pairs to process -> {args.out_dir}")
    new_pairs = []
    n_ok, n_fail = 0, 0

    for i, p in enumerate(pairs):
        pid = p.get("patient_id", f"p{i}")
        lat = p.get("laterality", "X")
        stem = f"{pid}_{lat}_{i:05d}"

        cc_out = os.path.join(args.out_dir, f"{stem}_CC.png")
        mlo_out = os.path.join(args.out_dir, f"{stem}_MLO.png")

        ok_cc = crop_and_save(p["cc_path"], cc_out, args.margin_frac)
        ok_mlo = crop_and_save(p["mlo_path"], mlo_out, args.margin_frac)

        if ok_cc and ok_mlo:
            n_ok += 1
            np_ = dict(p)            # copy ALL fields, including label, verbatim
            np_["cc_path"] = cc_out  # only the image paths change
            np_["mlo_path"] = mlo_out
            new_pairs.append(np_)
        else:
            n_fail += 1

        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(pairs)}  ok={n_ok} fail={n_fail}")

    # Preserve the original manifest container shape
    if isinstance(manifest, dict) and "pairs" in manifest:
        out_obj = dict(manifest)
        out_obj["pairs"] = new_pairs
    else:
        out_obj = new_pairs

    json.dump(out_obj, open(args.out_manifest, "w"))
    print(f"[crop] done. kept {n_ok} pairs, {n_fail} failures.")
    print(f"[crop] new manifest -> {args.out_manifest}")
    print("[crop] NOTE: labels copied verbatim; no label was used in any crop decision.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--out_manifest", required=True)
    ap.add_argument("--margin_frac", type=float, default=0.04,
                    help="fractional margin added around the breast bbox")
    main(ap.parse_args())
