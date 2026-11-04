"""Protocol P2 data: lesion-level manifest, inner splits, in-memory dataset."""

import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def file_md5(path):
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def inner_split(pairs, indices, val_fraction=0.15, seed=2026):
    """Patient-grouped, label-stratified split of `indices`.

    A patient is stratified by whether any of their lesions is malignant.
    The split depends on the outer fold only, never on the model seed.
    """
    by_patient = defaultdict(list)
    for i in indices:
        by_patient[pairs[i]["patient_id"]].append(i)
    patients = sorted(by_patient)
    strata = [int(any(pairs[i]["label"] for i in by_patient[p])) for p in patients]
    train_p, val_p = train_test_split(patients, test_size=val_fraction, stratify=strata, random_state=seed)
    train_idx = sorted(i for p in train_p for i in by_patient[p])
    val_idx = sorted(i for p in val_p for i in by_patient[p])
    assert not ({pairs[i]["patient_id"] for i in train_idx} & {pairs[i]["patient_id"] for i in val_idx})
    return train_idx, val_idx


def build_splits(manifest):
    """Return a dict: fold -> {train, inner_val, outer_val}. Fold -1 is the test phase."""
    pairs = manifest["pairs"]
    splits = {}
    for rec in manifest["folds"]:
        tr, iv = inner_split(pairs, rec["train_idx"])
        splits[rec["fold"]] = {"train": tr, "inner_val": iv, "outer_val": rec["val_idx"]}
    tr, iv = inner_split(pairs, list(range(len(pairs))))
    splits[-1] = {"train": tr, "inner_val": iv, "outer_val": []}
    return splits


def check_splits(manifest, splits):
    pairs = manifest["pairs"]
    pid = lambda idx: {pairs[i]["patient_id"] for i in idx}
    test_patients = {p["patient_id"] for p in manifest.get("test_pairs", [])}
    assert not (pid(range(len(pairs))) & test_patients), "development and test patients overlap"
    for fold, s in splits.items():
        assert not (pid(s["train"]) & pid(s["inner_val"])), f"fold {fold}: inner leakage"
        assert not (pid(s["train"]) & pid(s["outer_val"])), f"fold {fold}: outer leakage"
        assert not (pid(s["inner_val"]) & pid(s["outer_val"])), f"fold {fold}: inner/outer overlap"


def train_transform(img_size=224):
    return T.Compose([
        T.Resize((img_size + 20, img_size + 20)),
        T.RandomCrop((img_size, img_size)),
        T.RandomHorizontalFlip(0.5),
        T.RandomVerticalFlip(0.2),
        T.RandomRotation(20),
        T.ColorJitter(brightness=0.2, contrast=0.2),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def eval_transform(img_size=224):
    return T.Compose([T.Resize((img_size, img_size)), T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])


class ImageCache:
    """Decode every patch once. Kaggle input storage is slow per file."""

    def __init__(self, patch_dir):
        self.patch_dir = Path(patch_dir)
        self.images = {}

    def get(self, name):
        if name not in self.images:
            with Image.open(self.patch_dir / name) as im:
                self.images[name] = im.convert("RGB")
        return self.images[name]

    def preload(self, pairs):
        for p in pairs:
            self.get(p["cc"])
            self.get(p["mlo"])


class PairDataset(Dataset):
    def __init__(self, pairs, indices, cache, transform, paired_aug=True):
        self.pairs, self.indices, self.cache, self.transform = pairs, list(indices), cache, transform
        self.paired_aug = paired_aug

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]
        p = self.pairs[idx]
        cc, mlo = self.cache.get(p["cc"]), self.cache.get(p["mlo"])
        if self.paired_aug:
            state = torch.random.get_rng_state()
            cc = self.transform(cc)
            torch.random.set_rng_state(state)
            mlo = self.transform(mlo)
        else:
            cc, mlo = self.transform(cc), self.transform(mlo)
        return {"img_cc": cc, "img_mlo": mlo, "label": int(p["label"]), "index": idx}


def seed_worker(worker_id):
    np.random.seed(torch.initial_seed() % 2**32)


def make_loaders(manifest, split, cache, seed, batch_size=16, num_workers=2):
    pairs = manifest["pairs"]
    labels = [pairs[i]["label"] for i in split["train"]]
    counts = np.bincount(labels, minlength=2)
    weights = [1.0 / counts[l] for l in labels]
    gen = torch.Generator().manual_seed(seed)
    sampler = WeightedRandomSampler(weights, 2 * len(weights), replacement=True, generator=gen)
    common = dict(num_workers=num_workers, pin_memory=True, persistent_workers=num_workers > 0,
                  worker_init_fn=seed_worker)
    train = DataLoader(PairDataset(pairs, split["train"], cache, train_transform()), batch_size=batch_size,
                       sampler=sampler, drop_last=True, **common)
    loaders = {"train": train}
    for name in ("inner_val", "outer_val"):
        if split[name]:
            loaders[name] = DataLoader(PairDataset(pairs, split[name], cache, eval_transform()),
                                       batch_size=32, shuffle=False, **common)
    return loaders


def load_manifest(path):
    with open(path) as f:
        return json.load(f)
