#!/usr/bin/env python
"""Parameters, operations and per-pair latency on one GPU, batch size 1."""

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.baseline_models import mc_dropout_predict
from training.p2_config import MC_DROPOUT_PASSES
from training.train_p2 import build

DUAL_METHODS = ["cued_net", "tmc", "softmax_dual", "cross_view", "cued_convnext", "tmc_convnext", "softmax_convnext", "cross_view_convnext"]


def latency(fn, device, warmup=30, repeats=200):
    x = torch.rand(1, 3, 224, 224, device=device)
    times = []
    with torch.no_grad():
        for i in range(warmup + repeats):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t = time.perf_counter()
            fn(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            if i >= warmup:
                times.append((time.perf_counter() - t) * 1000)
    times = torch.tensor(times)
    return dict(mean_ms=times.mean().item(), std_ms=times.std().item(), median_ms=times.median().item())


def operations(model, device):
    try:
        from torch.utils.flop_counter import FlopCounterMode
        x = torch.rand(1, 3, 224, 224, device=device)
        with FlopCounterMode(model, display=False) as counter:
            model(x, x)
        return counter.get_total_flops()
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    result = {}
    for method in DUAL_METHODS + ["sv_edl"]:
        model, _ = build(method, pretrained=False)
        model = model.to(device).eval()
        call = (lambda x, m=model: m(x)) if method == "sv_edl" else (lambda x, m=model: m(x, x))
        result[method] = dict(params_m=sum(p.numel() for p in model.parameters()) / 1e6, flops=operations(model, device) if method != "sv_edl" else None,
                              latency=latency(call, device))
    for name, suffix in (("softmax_dual", ""), ("softmax_convnext", "_convnext")):
        base, _ = build(name, pretrained=False)
        base = base.to(device).eval()
        result["mc_dropout" + suffix] = dict(passes=MC_DROPOUT_PASSES, latency=latency(lambda x: mc_dropout_predict(base, x, x, T=MC_DROPOUT_PASSES), device, 5, 40))
        members = [build(name, pretrained=False)[0].to(device).eval() for _ in range(5)]
        result["deep_ensemble" + suffix] = dict(members=5, latency=latency(lambda x: [m(x, x) for m in members], device, 10, 100))
        del base, members
    result["gpu"] = gpu
    json.dump(result, open(args.out, "w"), indent=1)
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
