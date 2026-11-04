#!/usr/bin/env python
"""Locked evaluation of the held-out split. Runs once, for all methods together.

The script needs the commit hash of the analysis plan, and writes a lock file.
It refuses a second run unless --disclose_rerun gives the reason, which is
recorded and must appear in the paper.
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.p2_data import ImageCache, PairDataset, eval_transform, load_manifest
from training.p2_config import METHODS, SEEDS
from training.train_p2 import build, predict, write_csv

LOCKED = ["cued_net", "softmax_dual", "tmc", "sv_edl", "cross_view"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--patches", required=True)
    ap.add_argument("--runs", required=True, help="folder with fold_-1 checkpoints")
    ap.add_argument("--out", required=True)
    ap.add_argument("--plan_commit", required=True)
    ap.add_argument("--disclose_rerun", default="")
    args = ap.parse_args()

    out = Path(args.out)
    lock = out / "test_evaluation.lock.json"
    if lock.exists() and not args.disclose_rerun:
        raise SystemExit(f"already evaluated: {lock.read_text()}")

    manifest = load_manifest(args.manifest)
    pairs = manifest["test_pairs"]
    missing = [(m, s) for m in LOCKED for s in SEEDS
               if not (Path(args.runs) / m / "fold_-1" / f"seed_{s}" / "best_model.pt").exists()]
    assert not missing, f"train every locked model first: {missing}"
    assert not manifest["test_overlap"], "development and test patients overlap"

    cache = ImageCache(args.patches)
    cache.preload(pairs)
    loader = torch.utils.data.DataLoader(PairDataset(pairs, range(len(pairs)), cache, eval_transform()), batch_size=32, num_workers=2)
    device = torch.device("cuda")
    mc = {m: 50 if METHODS[m].get("mc_dropout") else 0 for m in LOCKED}
    for method in LOCKED:
        for seed in SEEDS:
            model, _ = build(method, pretrained=False)
            path = Path(args.runs) / method / "fold_-1" / f"seed_{seed}" / "best_model.pt"
            model.load_state_dict(torch.load(path, map_location="cpu", weights_only=False)["model_state_dict"])
            model = model.to(device)
            pred = predict(model, loader, device, mc[method])
            target = out / method / "fold_-1" / f"seed_{seed}"
            target.mkdir(parents=True, exist_ok=True)
            write_csv(target / "test.csv", pred, pairs, method, -1, seed, "test")
    lock.write_text(json.dumps(dict(plan_commit=args.plan_commit, manifest_md5=manifest["md5"], n_test=len(pairs),
                                    time=time.strftime("%Y-%m-%d %H:%M:%S"), rerun_reason=args.disclose_rerun)))


if __name__ == "__main__":
    main()
