"""Response of the three terms to controlled changes of the input. ANALYSIS_PLAN.md section 13.

Runs after evaluate_test.py, once, on the reference model of each arm.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.p2_data import ImageCache, PairDataset, eval_transform, load_manifest
from evaluation.decomposition import add_decomposition
from training.p2_config import SEEDS
from training.train_p2 import build, predict, write_csv

CONDITIONS = ("swap_opposite", "swap_same", "blur")
TERMS = ("u_amb", "u_ev", "u_vd", "u_total")


def swapped(pairs, same_label, seed=2026):
    """Give each case the MLO view of another patient, with the same or the opposite label."""
    rng = np.random.default_rng(seed)
    out = []
    for p in pairs:
        pool = [q for q in pairs if q["patient_id"] != p["patient_id"] and (q["label"] == p["label"]) == same_label]
        out.append(dict(p, mlo=pool[int(rng.integers(len(pool)))]["mlo"]))
    return out


def condition_loader(name, pairs, cache):
    if name == "blur":
        data, transform = pairs, T.Compose([eval_transform(), T.GaussianBlur(25, 3.0)])
    else:
        data, transform = swapped(pairs, name == "swap_same"), eval_transform()
    return torch.utils.data.DataLoader(PairDataset(data, range(len(data)), cache, transform), batch_size=32, num_workers=2)


def run(args):
    out = Path(args.out)
    assert (out / "test_evaluation.lock.json").exists(), "run evaluate_test.py first"
    lock = out / "term_response.lock.json"
    if lock.exists():
        raise SystemExit(f"already run: {lock.read_text()}")
    manifest = load_manifest(args.manifest)
    pairs = manifest["test_pairs"]
    cache = ImageCache(args.patches)
    cache.preload(pairs)
    device = torch.device("cuda")
    for method in args.methods.split(","):
        for seed in SEEDS:
            model, _ = build(method, pretrained=False)
            path = Path(args.runs) / method / "fold_-1" / f"seed_{seed}" / "best_model.pt"
            model.load_state_dict(torch.load(path, map_location="cpu", weights_only=False)["model_state_dict"])
            model = model.to(device)
            for name in CONDITIONS:
                pred = predict(model, condition_loader(name, pairs, cache), device)
                write_csv(out / method / "fold_-1" / f"seed_{seed}" / f"{name}.csv", pred, pairs, method, -1, seed, name)
    lock.write_text(json.dumps(dict(methods=args.methods, time=time.strftime("%Y-%m-%d %H:%M:%S"))))


def per_case(folder, name):
    d = add_decomposition(pd.concat(pd.read_csv(f) for f in sorted(Path(folder).glob(f"seed_*/{name}.csv"))))
    return d.groupby(["pair_key", "patient_id"])[list(TERMS)].mean()


def summarise(out, method, B=10000, seed=20261105):
    folder = Path(out) / method / "fold_-1"
    clean = per_case(folder, "test")
    group = pd.factorize(clean.index.get_level_values("patient_id"))[0]
    members = [np.where(group == g)[0] for g in range(group.max() + 1)]
    rng = np.random.default_rng(seed)
    draws = [np.concatenate([members[i] for i in rng.integers(len(members), size=len(members))]) for _ in range(B)]
    result = dict(clean={t: float(clean[t].mean()) for t in TERMS})
    for name in CONDITIONS:
        delta = (per_case(folder, name).reindex(clean.index) - clean).values
        boot = np.array([delta[d].mean(0) for d in draws])
        result[name] = {t: dict(change=float(delta[:, i].mean()), ci95=np.percentile(boot[:, i], [2.5, 97.5]).tolist()) for i, t in enumerate(TERMS)}
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest")
    ap.add_argument("--patches")
    ap.add_argument("--runs")
    ap.add_argument("--out", required=True)
    ap.add_argument("--methods", default="cued_net,cued_convnext_k5")
    ap.add_argument("--summary", default="")
    args = ap.parse_args()
    if args.summary:
        result = {m: summarise(args.out, m) for m in args.methods.split(",")}
        Path(args.summary).write_text(json.dumps(result, indent=1))
    else:
        run(args)


if __name__ == "__main__":
    main()
