#!/usr/bin/env python3
"""Benchmark sweep: both models x {1, 4, 16} MiB x {overlap, overlap + swap, direct}, in one Slurm allocation.

  python scripts/sweep.py --nodes 4 --account <slurm account> [--qos Q] [--minutes N] [--gpu-plan 0|1] [--out DIR]

Writes DIR/sweep_<N>n.csv (per-rank, per-iteration metrics) and prints one result line per cell.
"""
import argparse
import datetime
import os
import re
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nodes", type=int, default=4)
    ap.add_argument("--account", default=os.environ.get("SLURM_ACCOUNT", ""))
    ap.add_argument("--qos", default="")
    ap.add_argument("--minutes", type=int, default=0)
    ap.add_argument("--constraint", default="gpu", help="salloc -C")
    ap.add_argument("--gpus-per-node", type=int, default=4)
    ap.add_argument("--gpu-plan", type=int, default=1)
    ap.add_argument("--out", default=os.path.join("runs", datetime.date.today().isoformat()))
    args = ap.parse_args()
    if not os.environ.get("ZEPP_CONDA_ENV"):
        sys.exit("set ZEPP_CONDA_ENV to the conda env with PyTorch (see README: Build)")
    if not args.account:
        sys.exit("pass --account <slurm account> (or set SLURM_ACCOUNT)")
    n = args.nodes
    qos = args.qos or ("interactive" if n <= 4 else "regular")
    minutes = args.minutes or (45 if n <= 4 else 15 if n <= 8 else 30)
    router_c = 0.5 if n >= 16 else 0.25
    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)
    csv_path = os.path.join(out, f"sweep_{n}n.csv")

    print(f"allocating {n} nodes ({qos}, {minutes} min) ...", flush=True)
    salloc_log = os.path.join(out, f"salloc_{n}n.log")
    with open(salloc_log, "w") as f:
        subprocess.run(["salloc", "-q", qos, "-C", args.constraint, "-A", args.account, "-N", str(n),
                        f"--gpus-per-node={args.gpus_per_node}", "--no-shell", "-t", str(minutes)],
                       stdout=f, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
    m = re.search(r"Granted job allocation (\d+)", open(salloc_log).read())
    if not m:
        sys.exit("allocation failed: " + open(salloc_log).read())
    job = m.group(1)
    try:
        while "ready for job" not in open(salloc_log).read():
            time.sleep(5)
        for model in ("qwen3", "k2"):
            for budget in (1, 4, 16):
                for strat, swap in (("overlap", 0), ("overlap", 1), ("direct", 0)):
                    if strat == "direct" and n >= 16 and budget >= 16:
                        print(f"[{time.strftime('%T')}] {model} b{budget} direct: skipped at {n} nodes", flush=True)
                        continue
                    log = os.path.join(out, f"{model}_b{budget}_{strat}_swap{swap}_{n}n.log")
                    print(f"[{time.strftime('%T')}] {model} b{budget} {strat} swap={swap}", flush=True)
                    cmd = (f"source '{ROOT}/env/setup.sh' && cd '{ROOT}' && bench/launch.sh bench/replay.py "
                           f"--model {model} --budget-mib {budget} --comm-strategy {strat} --swap {swap} "
                           f"--router-c {router_c} --gpu-plan {args.gpu_plan} --iters 10 --warmup 5 --out '{csv_path}'")
                    with open(log, "w") as f:
                        rc = subprocess.run(["timeout", "300", "srun", f"--jobid={job}", f"--nodes={n}",
                                             "--ntasks-per-node=1", f"--gpus-per-node={args.gpus_per_node}",
                                             "bash", "-lc", cmd],
                                            stdout=f, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL).returncode
                    lines = [l for l in open(log, errors="replace") if l.startswith("[result]")]
                    print("".join(lines) if lines else f"    cell failed (rc {rc}), see {log}", end="" if lines else "\n",
                          flush=True)
    finally:
        subprocess.run(["scancel", job])
    print(f"results: {csv_path}")


if __name__ == "__main__":
    main()
