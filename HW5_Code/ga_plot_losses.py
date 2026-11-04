#!/usr/bin/env python3
"""
Plot train/val loss curves for llm.c runs and report the lowest val loss.

Usage:
    python plot_losses.py RUN_NAME [RUN_NAME ...] [--log_dir my_logs] [--out_dir plots]
    python plot_losses.py --log_dir my_logs          # no run names -> every folder in log_dir

Each run is expected at {log_dir}/{run_name}/main.log
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")  # no display needed, just write PNGs
import matplotlib.pyplot as plt
import numpy as np


def parse_logfile(logfile):
    # The job could crash and restart, re-logging older steps, so keep a
    # dict per stream and let newer entries overwrite older ones.
    streams = {}  # stream:str -> {step: val}
    with open(logfile, "r") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                step = int(parts[0].split(":")[1])
                stream, val = parts[1].split(":")
                val = float(val)
            except (IndexError, ValueError):
                continue
            streams.setdefault(stream, {})[step] = val

    streams_xy = {}
    for k, v in streams.items():
        xy = sorted(v.items())
        xs, ys = zip(*xy)
        streams_xy[k] = (np.array(xs), np.array(ys))
    return streams_xy


def smooth_moving_average(signal, window_size):
    if signal.ndim != 1:
        raise ValueError("smooth_moving_average only accepts 1D arrays.")
    if signal.size < window_size:
        raise ValueError("Input vector needs to be bigger than window size.")
    if window_size < 3:
        return signal
    s = np.pad(signal, (window_size // 2, window_size - 1 - window_size // 2), mode="edge")
    w = np.ones(window_size) / window_size
    return np.convolve(s, w, mode="valid")


def plot_run(run_name, streams, out_dir, smooth=0):
    plt.figure(figsize=(8, 6))

    if "trl" in streams:
        xs, ys = streams["trl"]
        if smooth and ys.size >= smooth:
            ys = smooth_moving_average(ys, smooth)
        plt.plot(xs, ys, marker="o", markersize=2, label=f"llm.c ({run_name}) train loss")

    if "tel" in streams:
        xs, ys = streams["tel"]
        plt.plot(xs, ys, marker="o", label=f"llm.c ({run_name}) val loss")

    plt.xlabel("steps")
    plt.ylabel("loss")
    plt.yscale("log")
    plt.legend()
    plt.title(f"Loss: {run_name}")
    plt.tight_layout()

    out_path = os.path.join(out_dir, f"{run_name}.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="*", help="run names (folders inside log_dir)")
    parser.add_argument("--log_dir", default="my_logs", help="directory containing run folders")
    parser.add_argument("--out_dir", default="plots", help="where to save PNGs")
    parser.add_argument("--smooth", type=int, default=0, help="moving-average window for train loss (0 = off)")
    args = parser.parse_args()

    runs = args.runs
    if not runs:
        runs = sorted(d for d in os.listdir(args.log_dir)
                      if os.path.isdir(os.path.join(args.log_dir, d)))
    if not runs:
        sys.exit(f"No runs found in {args.log_dir}")

    os.makedirs(args.out_dir, exist_ok=True)

    results = []  # (run_name, min_val_loss, step)
    for run_name in runs:
        logfile = os.path.join(args.log_dir, run_name, "main.log")
        if not os.path.isfile(logfile):
            print(f"[skip] {run_name}: {logfile} not found")
            continue

        streams = parse_logfile(logfile)
        out_path = plot_run(run_name, streams, args.out_dir, args.smooth)

        if "tel" in streams:
            xs, ys = streams["tel"]
            i = int(np.argmin(ys))
            results.append((run_name, float(ys[i]), int(xs[i])))
        else:
            results.append((run_name, float("nan"), -1))
        print(f"[saved] {out_path}")

    # Summary table, best run first
    print("\nLowest validation loss per run:")
    print(f"{'run_name':<40} {'min_val_loss':>12} {'at_step':>8}")
    print("-" * 62)
    for run_name, loss, step in sorted(results, key=lambda r: (np.isnan(r[1]), r[1])):
        loss_str = f"{loss:.6f}" if not np.isnan(loss) else "n/a"
        print(f"{run_name:<40} {loss_str:>12} {step:>8}")


if __name__ == "__main__":
    main()
