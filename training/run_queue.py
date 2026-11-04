#!/usr/bin/env python
"""Run many train_p2 jobs across the available GPUs, resuming finished ones.

A job counts as finished when done.json exists under --out or any --resume dir.
Finished folders from earlier Kaggle sessions are copied in, so the output of
each session is cumulative. The queue stops starting jobs at --hours.
"""

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from training.p2_config import OUTER_FOLDS, SEEDS, TEST_PHASE_FOLD


def expand(spec):
    """'cued_net:cv,cued_net:test' -> list of (method, fold, seed)."""
    jobs = []
    for item in spec.split(","):
        method, phase = item.split(":")
        folds = {"cv": OUTER_FOLDS, "test": [TEST_PHASE_FOLD]}[phase]
        jobs += [(method, f, s) for s in SEEDS for f in folds]
    return jobs


def restore(out, resume_dirs):
    for r in resume_dirs:
        for done in Path(r).glob("**/done.json"):
            rel = done.parent.relative_to(done.parents[3])
            dst = Path(out) / rel
            if not (dst / "done.json").exists():
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copytree(done.parent, dst, dirs_exist_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--patches", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--resume", nargs="*", default=[])
    ap.add_argument("--gpus", default="0,1")
    ap.add_argument("--hours", type=float, default=11.0)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--extra", default="", help="extra train_p2 arguments")
    args = ap.parse_args()

    restore(args.out, args.resume)
    todo = [j for j in expand(args.jobs)
            if not (Path(args.out) / j[0] / f"fold_{j[1]}" / f"seed_{j[2]}" / "done.json").exists()]
    print(f"{len(todo)} jobs to run", flush=True)
    start = time.time()
    running = {}
    gpus = args.gpus.split(",")
    while todo or running:
        for gpu in gpus:
            if gpu in running or not todo or time.time() - start > args.hours * 3600:
                continue
            method, fold, seed = todo.pop(0)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)
            log = Path(args.out) / "logs" / f"{method}_f{fold}_s{seed}.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            cmd = [sys.executable, str(ROOT / "training" / "train_p2.py"), "--method", method, "--fold", str(fold),
                   "--seed", str(seed), "--manifest", args.manifest, "--patches", args.patches, "--out", args.out,
                   "--workers", str(args.workers)] + args.extra.split()
            running[gpu] = (subprocess.Popen(cmd, env=env, stdout=open(log, "w"), stderr=subprocess.STDOUT), (method, fold, seed), log)
        for gpu, (proc, job, log) in list(running.items()):
            code = proc.poll()
            if code is not None:
                print(f"{job} exit {code} at {(time.time() - start) / 3600:.2f} h", flush=True)
                if code != 0:
                    print(open(log).read()[-1500:], flush=True)
                del running[gpu]
        if not running and (not todo or time.time() - start > args.hours * 3600):
            break
        time.sleep(5)
    left = [j for j in expand(args.jobs)
            if not (Path(args.out) / j[0] / f"fold_{j[1]}" / f"seed_{j[2]}" / "done.json").exists()]
    print(f"finished. {len(left)} jobs left", flush=True)


if __name__ == "__main__":
    main()
