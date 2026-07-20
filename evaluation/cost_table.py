#!/usr/bin/env python
"""
cost_table.py — Computational cost table for R2.10 / Table V.

Measures per model: parameters (M), GMACs per inference, peak GPU memory (MB),
inference latency per CC-MLO pair (ms, warm), throughput (pairs/s).
No retraining — instantiates architectures and times forward passes.

Derived costs:
  MC-Dropout  = single forward x T (T=50 stochastic passes at inference)
  Deep-Ensemble = single forward x M (M=5 models)

GPU recommended (RTX 3090). Run on the SAME pod for hardware symmetry.
"""
import argparse, json, time, subprocess, sys
from pathlib import Path
import numpy as np
import torch

# ensure thop present
try:
    from thop import profile as thop_profile
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "thop",
                    "--quiet", "--root-user-action=ignore"], check=True)
    from thop import profile as thop_profile

from models.cued_net import CUEDNet
from baseline_models_edl_tmc import SingleViewEDL, TMCDualView

T_MCDROPOUT = 50   # stochastic passes
M_ENSEMBLE  = 5    # ensemble members

def count_params(model):
    return sum(p.numel() for p in model.parameters())

def measure_latency(model, inputs, device, n_warmup=10, n_trials=100):
    model.eval()
    with torch.no_grad():
        for _ in range(n_warmup):
            model(*inputs)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_trials):
            model(*inputs)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
    return (t1 - t0) / n_trials * 1000.0   # ms per pair

def peak_memory_mb(model, inputs, device):
    if device.type != "cuda":
        return float("nan")
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    model.eval()
    with torch.no_grad():
        model(*inputs)
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / (1024**2)

def gmacs_dualview(model, cc, mlo):
    """thop profile for dual-view forward (cc, mlo)."""
    try:
        macs, _ = thop_profile(model, inputs=(cc, mlo), verbose=False)
        return macs / 1e9
    except Exception as e:
        print(f"    [gmacs dual-view failed: {e}]")
        return float("nan")

def gmacs_singleview(model, x):
    try:
        macs, _ = thop_profile(model, inputs=(x,), verbose=False)
        return macs / 1e9
    except Exception as e:
        print(f"    [gmacs single-view failed: {e}]")
        return float("nan")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/workspace/cued_net/calib_out")
    ap.add_argument("--batch", type=int, default=1, help="inference batch (1 pair)")
    args = ap.parse_args()
    Path(args.out).mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # dummy inputs matching the 224x224 RGB dual-view interface
    cc  = torch.randn(args.batch, 3, 224, 224, device=device)
    mlo = torch.randn(args.batch, 3, 224, 224, device=device)
    single = torch.randn(args.batch, 3, 224, 224, device=device)

    results = {}

    # ---- CUED-Net (single model) ----
    print("\n[CUED-Net]")
    cued = CUEDNet(num_classes=2, pretrained=False).to(device)
    p_cued = count_params(cued)
    g_cued = gmacs_dualview(cued, cc, mlo)
    mem_cued = peak_memory_mb(cued, (cc, mlo), device)
    lat_cued = measure_latency(cued, (cc, mlo), device)
    results["CUED-Net"] = {
        "params_M": p_cued/1e6, "gmacs": g_cued, "mem_MB": mem_cued,
        "latency_ms": lat_cued, "throughput_pairs_s": 1000.0/lat_cued,
        "passes": 1,
    }
    print(f"  params={p_cued/1e6:.2f}M  GMACs={g_cued:.2f}  mem={mem_cued:.0f}MB  "
          f"lat={lat_cued:.2f}ms  thr={1000/lat_cued:.1f}/s")
    del cued; torch.cuda.empty_cache() if device.type=="cuda" else None

    # ---- TMC dual-view ----
    print("\n[TMC]")
    try:
        tmc = TMCDualView(num_classes=2).to(device)
    except TypeError:
        tmc = TMCDualView().to(device)
    p_tmc = count_params(tmc)
    g_tmc = gmacs_dualview(tmc, cc, mlo)
    mem_tmc = peak_memory_mb(tmc, (cc, mlo), device)
    lat_tmc = measure_latency(tmc, (cc, mlo), device)
    results["TMC"] = {
        "params_M": p_tmc/1e6, "gmacs": g_tmc, "mem_MB": mem_tmc,
        "latency_ms": lat_tmc, "throughput_pairs_s": 1000.0/lat_tmc, "passes": 1,
    }
    print(f"  params={p_tmc/1e6:.2f}M  GMACs={g_tmc:.2f}  mem={mem_tmc:.0f}MB  "
          f"lat={lat_tmc:.2f}ms  thr={1000/lat_tmc:.1f}/s")
    del tmc; torch.cuda.empty_cache() if device.type=="cuda" else None

    # ---- Single-view EDL ----
    print("\n[Single-view-EDL]")
    try:
        edl = SingleViewEDL(num_classes=2).to(device)
    except TypeError:
        edl = SingleViewEDL().to(device)
    p_edl = count_params(edl)
    g_edl = gmacs_singleview(edl, single)
    mem_edl = peak_memory_mb(edl, (single,), device)
    lat_edl = measure_latency(edl, (single,), device)
    results["Single-view-EDL"] = {
        "params_M": p_edl/1e6, "gmacs": g_edl, "mem_MB": mem_edl,
        "latency_ms": lat_edl, "throughput_pairs_s": 1000.0/lat_edl, "passes": 1,
    }
    print(f"  params={p_edl/1e6:.2f}M  GMACs={g_edl:.2f}  mem={mem_edl:.0f}MB  "
          f"lat={lat_edl:.2f}ms  thr={1000/lat_edl:.1f}/s")

    # ---- MC-Dropout: single-view EDL backbone x T passes ----
    # (uses the single-model base cost; latency scales by T)
    print(f"\n[MC-Dropout]  (= base x T={T_MCDROPOUT} passes)")
    results["MC-Dropout"] = {
        "params_M": p_edl/1e6, "gmacs": g_edl*T_MCDROPOUT, "mem_MB": mem_edl,
        "latency_ms": lat_edl*T_MCDROPOUT,
        "throughput_pairs_s": 1000.0/(lat_edl*T_MCDROPOUT), "passes": T_MCDROPOUT,
    }
    print(f"  params={p_edl/1e6:.2f}M  GMACs={g_edl*T_MCDROPOUT:.2f}  "
          f"lat={lat_edl*T_MCDROPOUT:.2f}ms  ({T_MCDROPOUT} passes)")
    del edl; torch.cuda.empty_cache() if device.type=="cuda" else None

    # ---- Deep-Ensemble: CUED-Net x M models ----
    print(f"\n[Deep-Ensemble]  (= CUED-Net x M={M_ENSEMBLE} models)")
    results["Deep-Ensemble(M=5)"] = {
        "params_M": p_cued/1e6*M_ENSEMBLE, "gmacs": g_cued*M_ENSEMBLE,
        "mem_MB": mem_cued,  # sequential inference: peak mem ~ one model
        "latency_ms": lat_cued*M_ENSEMBLE,
        "throughput_pairs_s": 1000.0/(lat_cued*M_ENSEMBLE), "passes": M_ENSEMBLE,
    }
    print(f"  params={p_cued/1e6*M_ENSEMBLE:.2f}M  GMACs={g_cued*M_ENSEMBLE:.2f}  "
          f"lat={lat_cued*M_ENSEMBLE:.2f}ms  ({M_ENSEMBLE} models)")

    json.dump(results, open(Path(args.out)/"cost_table.json","w"), indent=2)
    print(f"\n[ok] -> {Path(args.out)/'cost_table.json'}")

    # summary table
    print("\n── COST TABLE (Table V) ──")
    print(f"{'Model':22s} {'Params(M)':>10s} {'GMACs':>8s} {'Mem(MB)':>8s} "
          f"{'Lat(ms)':>9s} {'Thr(/s)':>9s} {'Passes':>7s}")
    order = ["Single-view-EDL","CUED-Net","TMC","MC-Dropout","Deep-Ensemble(M=5)"]
    for name in order:
        r = results[name]
        print(f"{name:22s} {r['params_M']:10.2f} {r['gmacs']:8.2f} "
              f"{r['mem_MB']:8.0f} {r['latency_ms']:9.2f} "
              f"{r['throughput_pairs_s']:9.1f} {r['passes']:7d}")

if __name__ == "__main__":
    main()