"""Response of the three terms to controlled changes of the input. ANALYSIS_PLAN.md section 13.

--development runs the changes on the outer folds with the fold checkpoints of cued_net, as a
pilot of the design. The default mode runs once after evaluate_test.py, on the held-out split.
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
from data.p2_data import ImageCache, PairDataset, build_splits, eval_transform, load_manifest
from evaluation.decomposition import add_decomposition
from training.p2_config import SEEDS
from training.train_p2 import build, predict, write_csv

SWAPS = {"swap_opposite": False, "swap_same": True}
SWAP_DRAWS = (2026, 2027, 2028)
BLUR = {"blur_1p5": 1.5, "blur_3": 3.0, "blur_6": 6.0}
OCCLUDE = {"occlude_25": 0.25, "occlude_50": 0.50}
CONDITIONS = tuple(SWAPS) + tuple(BLUR) + tuple(OCCLUDE)
TERMS = ("u_amb", "u_ev", "u_vd", "u_total", "share_evid", "share_vd")


def swapped(pairs, same_label, seed=2026):
    """Give each case the MLO view of another patient, with the same or the opposite label."""
    rng = np.random.default_rng(seed)
    out = []
    for p in pairs:
        pool = [q for q in pairs if q["patient_id"] != p["patient_id"] and (q["label"] == p["label"]) == same_label]
        out.append(dict(p, mlo=pool[int(rng.integers(len(pool)))]["mlo"]))
    return out


class Occlude:
    """Fill the central square with the ImageNet mean. Lesions sit at the center of the patch."""

    def __init__(self, fraction):
        self.fraction = fraction

    def __call__(self, x):
        side = int(self.fraction * x.shape[-1])
        start = (x.shape[-1] - side) // 2
        x = x.clone()
        x[..., start:start + side, start:start + side] = 0
        return x


def run_conditions(model, pairs, cache, device, folder, method, fold, seed):
    jobs = [(f"{n}_{d}", swapped(pairs, same, d), eval_transform()) for n, same in SWAPS.items() for d in SWAP_DRAWS]
    jobs += [(n, pairs, T.Compose([eval_transform(), T.GaussianBlur(25, s)])) for n, s in BLUR.items()]
    jobs += [(n, pairs, T.Compose([eval_transform(), Occlude(f)])) for n, f in OCCLUDE.items()]
    folder.mkdir(parents=True, exist_ok=True)
    for name, data, transform in jobs:
        loader = torch.utils.data.DataLoader(PairDataset(data, range(len(data)), cache, transform), batch_size=32, num_workers=2)
        write_csv(folder / f"{name}.csv", predict(model, loader, device), pairs, method, fold, seed, name)


def load(method, path, device):
    model, _ = build(method, pretrained=False)
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=False)["model_state_dict"])
    return model.to(device)


def run(args):
    out = Path(args.out)
    manifest = load_manifest(args.manifest)
    cache = ImageCache(args.patches)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.development:
        splits = build_splits(manifest)
        cache.preload(manifest["pairs"])
        for fold in range(5):
            pairs = [manifest["pairs"][i] for i in splits[fold]["outer_val"]]
            for seed in SEEDS:
                model = load("cued_net", Path(args.runs) / "cued_net" / f"fold_{fold}" / f"seed_{seed}" / "best_model.pt", device)
                run_conditions(model, pairs, cache, device, out / "cued_net" / f"fold_{fold}" / f"seed_{seed}", "cued_net", fold, seed)
        return
    assert (out / "test_evaluation.lock.json").exists(), "run evaluate_test.py first"
    lock = out / "term_response.lock.json"
    if lock.exists():
        raise SystemExit(f"already run: {lock.read_text()}")
    pairs = manifest["test_pairs"]
    cache.preload(pairs)
    for method in args.methods.split(","):
        for seed in SEEDS:
            model = load(method, Path(args.runs) / method / "fold_-1" / f"seed_{seed}" / "best_model.pt", device)
            run_conditions(model, pairs, cache, device, out / method / "fold_-1" / f"seed_{seed}", method, -1, seed)
    lock.write_text(json.dumps(dict(methods=args.methods, time=time.strftime("%Y-%m-%d %H:%M:%S"))))


def per_case(folder, name):
    files = sorted(Path(folder).glob(f"fold_*/seed_*/{name}.csv"))
    assert files, (folder, name)
    d = add_decomposition(pd.concat(pd.read_csv(f) for f in files))
    d["share_evid"] = d.u_ev / (d.u_amb + d.u_ev)
    d["share_vd"] = d.u_vd / d.u_total
    return d.groupby(["pair_key", "patient_id"])[list(TERMS)].mean()


def summarise(clean_folder, changed_folder, clean_name, B=10000, seed=20261105):
    clean = per_case(clean_folder, clean_name)
    group = pd.factorize(clean.index.get_level_values("patient_id"))[0]
    members = [np.where(group == g)[0] for g in range(group.max() + 1)]
    rng = np.random.default_rng(seed)
    draws = [np.concatenate([members[i] for i in rng.integers(len(members), size=len(members))]) for _ in range(B)]
    result = dict(clean={t: float(clean[t].mean()) for t in TERMS})
    for name in CONDITIONS:
        pattern = f"{name}_*" if name in SWAPS else name
        delta = (per_case(changed_folder, pattern).reindex(clean.index) - clean).values
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
    ap.add_argument("--development", action="store_true")
    ap.add_argument("--summary", default="")
    args = ap.parse_args()
    if args.summary and args.development:
        Path(args.summary).write_text(json.dumps(dict(cued_net=summarise(Path(args.runs) / "cued_net", Path(args.out) / "cued_net", "outer_val")), indent=1))
    elif args.summary:
        result = {m: summarise(Path(args.out) / m, Path(args.out) / m, "test") for m in args.methods.split(",")}
        Path(args.summary).write_text(json.dumps(result, indent=1))
    else:
        run(args)


if __name__ == "__main__":
    main()
