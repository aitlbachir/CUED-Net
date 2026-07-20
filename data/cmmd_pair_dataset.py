"""
CMMDPairDataset — CMMD External Validation Loader for CUED-Net  (v4)
====================================================================
CORRECTED to mirror the EXACT CBIS-DDSM test/val preprocessing used to train
CUED-Net (see cued_net/data/datasets.py :: CBISDDSMDataset._get_transforms):

    T.Resize((224, 224))
    T.ToTensor()
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])   # ImageNet

Previous versions used Normalize(0.5, 0.5) — WRONG. That mismatch would have
conflated a preprocessing artifact with genuine domain shift (the exact silent
bug flagged in the session summary). Fixed here.

Channel handling: CBIS used PIL .convert('RGB') on grayscale PNGs (→ 3 identical
channels). We replicate the single DICOM channel to 3 BEFORE ImageNet norm so
each channel gets its correct per-channel mean/std — matching training exactly.

DICOM concerns (CMMD-specific): VOI-LUT windowing, MONOCHROME1 inversion,
12/16-bit → 8-bit rescaling. Handled in _dicom_to_uint8.
"""

import json
from pathlib import Path

import numpy as np
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T


# ---------------------------------------------------------------------------
def _dicom_to_uint8(dcm: pydicom.Dataset) -> np.ndarray:
    """DICOM pixel array → uint8 [0,255], windowing + photometric handled."""
    try:
        img = apply_voi_lut(dcm.pixel_array.astype(np.float64), dcm, prefer_lut=True)
    except Exception:
        img = dcm.pixel_array.astype(np.float64)

    pmi = getattr(dcm, "PhotometricInterpretation", "MONOCHROME2").strip()
    if pmi == "MONOCHROME1":
        img = img.max() - img

    lo, hi = img.min(), img.max()
    img = (img - lo) / (hi - lo) * 255.0 if hi > lo else np.zeros_like(img)
    return img.astype(np.uint8)


# ---------------------------------------------------------------------------
def cmmd_collate_fn(batch):
    """Stack tensors; keep meta as list[dict] (None-safe)."""
    return {
        "img_cc":  torch.stack([b["img_cc"]  for b in batch]),
        "img_mlo": torch.stack([b["img_mlo"] for b in batch]),
        "label":   torch.stack([b["label"]   for b in batch]),
        "meta":    [b["meta"] for b in batch],
    }


# ---------------------------------------------------------------------------
class CMMDPairDataset(Dataset):
    """
    CMMD CC+MLO breast pairs for CUED-Net external validation.

    __getitem__ returns:
        img_cc  : (3, 224, 224) float32, ImageNet-normalised
        img_mlo : (3, 224, 224) float32, ImageNet-normalised
        label   : scalar int64  {0=benign, 1=malignant}
        meta    : dict (patient_id, laterality, subtype, age, abnormality)

    Key names (img_cc/img_mlo) match the CUED-Net training batch dict exactly.
    """

    # EXACT CBIS-DDSM eval transform (ImageNet normalisation)
    _TRANSFORM = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    def __init__(self, json_path, path_prefix_override=None):
        self.prefix_override = path_prefix_override
        with open(json_path, "r") as fh:
            raw = json.load(fh)
        required = {"patient_id", "laterality", "label", "cc_path", "mlo_path"}
        for i, item in enumerate(raw):
            missing = required - item.keys()
            if missing:
                raise ValueError(f"Record {i} missing fields: {missing}")
        self.records = raw

    def __len__(self):
        return len(self.records)

    def _resolve_path(self, p):
        if self.prefix_override is None:
            return Path(p)
        parts = Path(p).parts
        for i, part in enumerate(parts):
            if "cmmd" in part.lower():
                tail = Path(*parts[i + 1:]) if i + 1 < len(parts) else Path(parts[-1])
                return Path(self.prefix_override) / tail
        return Path(self.prefix_override) / Path(p).name

    def _load_view(self, path):
        resolved = self._resolve_path(path)
        if not resolved.exists():
            raise FileNotFoundError(
                f"View not found: {resolved} (manifest: {path}). "
                f"Tip: pass path_prefix_override='/workspace/cmmd_full'"
            )
        suffix = resolved.suffix.lower()
        if suffix in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"):
            # Pre-extracted 8-bit image (e.g. cropped PNGs). Just RGB-ify.
            return Image.open(str(resolved)).convert("RGB")
        # DICOM branch (unchanged)
        dcm = pydicom.dcmread(str(resolved), force=True)
        arr = _dicom_to_uint8(dcm)                       # (H,W) uint8
        # → PIL grayscale → RGB (3 identical channels), matching CBIS .convert('RGB')
        return Image.fromarray(arr).convert("RGB")

    def __getitem__(self, idx):
        rec = self.records[idx]
        img_cc  = self._TRANSFORM(self._load_view(rec["cc_path"]))
        img_mlo = self._TRANSFORM(self._load_view(rec["mlo_path"]))
        return {
            "img_cc":  img_cc,
            "img_mlo": img_mlo,
            "label":   torch.tensor(rec["label"], dtype=torch.long),
            "meta": {
                "patient_id":  rec["patient_id"],
                "laterality":  rec["laterality"],
                "abnormality": rec.get("abnormality"),
                "subtype":     rec.get("subtype"),
                "age":         rec.get("age"),
            },
        }


# ---------------------------------------------------------------------------
def build_cmmd_loader(json_path, batch_size=32, num_workers=4,
                      path_prefix_override=None):
    dataset = CMMDPairDataset(json_path, path_prefix_override=path_prefix_override)
    print(f"[CMMDPairDataset] {len(dataset)} pairs "
          f"| mal {sum(r['label'] for r in dataset.records)} "
          f"| ben {sum(1 - r['label'] for r in dataset.records)}")
    return DataLoader(dataset, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers,
                      pin_memory=torch.cuda.is_available(),
                      drop_last=False, collate_fn=cmmd_collate_fn)


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    jp = sys.argv[1] if len(sys.argv) > 1 else "/workspace/cued_net/cmmd_pairs_full.json"
    pref = sys.argv[2] if len(sys.argv) > 2 else None
    loader = build_cmmd_loader(jp, batch_size=4, num_workers=0, path_prefix_override=pref)
    b = next(iter(loader))
    assert b["img_cc"].shape == (4, 3, 224, 224), b["img_cc"].shape
    print("OK", b["img_cc"].shape, "labels", b["label"].tolist())
    print("meta[0]", b["meta"][0])
