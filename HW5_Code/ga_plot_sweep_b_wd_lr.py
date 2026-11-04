#!/usr/bin/env python3
"""
Analyze batch size x weight decay and batch size x learning rate sweeps from llm.c runs.

Expects run folders named like  B{batch_size}_LRp{lr_digits}_WD_{wd_label}, each with main.log:
    B8192_LRp0006_WD_p025 -> B=8192, lr=0.0006, wd=0.025
    B8192_LRp0024_WD_p1   -> B=8192, lr=0.0024, wd=0.1
    B8192_LRp0006_WD_1p6  -> B=8192, lr=0.0006, wd=1.6

Plots (saved to --out_dir):
    sweeping_weight_decay.png   val loss vs AdamW timescale tau = 1/(wd*lr*T), runs with lr == --wd_sweep_lr
    sweeping_learning_rate.png  val loss vs AdamW timescale tau, runs with wd == --lr_sweep_wd
    optimal_hp.png              optimal weight decay and optimal learning rate vs batch size

Usage:
    python ga_plot_sweep.py --log_dir my_logs
    python ga_plot_sweep.py --log_dir my_logs --metric last   # mean of last 5 evals instead of min
"""

import argparse
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

MODEL = "mup_tiniest_1L"
PARAMS = 3332096

MODEL_PARAMS = {
    "mup_tiniest_1L": 3332096,
    "mup_tinier_2L": 6960768,
    "mup_tiny_3L": 11180928,
}

TOTAL_TOKENS = (
    20 * PARAMS
)  # step_size = TOTAL_TOKENS / batch_size, as in the sweep scripts

RUN_RE = re.compile(r"^B(\d+)_LR(p{0,2}\d*p?\d+)_WD_?(\d*p?\d+)$")


def parse_p_float(s: str) -> float:
    if not s:
        return 0.0
    if "p" in s:
        # Converts '1p6' -> 1.6, 'p1' -> 0.1, 'p0006' -> 0.0006
        return float(s.replace("p", "."))
    if s.startswith("0") and not s.startswith("0."):
        # Converts zero-padded numbers like '0003' -> 0.0003, '0192' -> 0.0192
        return float(f"0.{s}")
    return float(s)


# ----------------------------------------------------------------------------- parsing


def parse_logfile(logfile):
    # Later entries for the same step overwrite earlier ones (handles restarts).
    streams = {}
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
    return {
        k: tuple(map(np.array, zip(*sorted(v.items())))) for k, v in streams.items()
    }


def val_metric(val, metric, last_k):
    if metric == "min":
        finite = val[np.isfinite(val)]
        return float(finite.min()) if finite.size else float("nan")
    return float(np.mean(val[-last_k:]))  # "last": NaN if the run diverged near the end


def collect_runs(log_dir, metric, last_k, total_tokens):
    """Return a list of dicts: {name, bs, lr, wd, steps, tau, loss}."""
    runs = []
    for name in sorted(os.listdir(log_dir)):
        m = RUN_RE.match(name)
        if not m:
            continue
        logfile = os.path.join(log_dir, name, "main.log")
        if not os.path.isfile(logfile):
            continue

        bs = int(m.group(1))
        lr = parse_p_float(m.group(2).replace("pp", "p"))
        wd = parse_p_float(m.group(3))
        steps = total_tokens // bs

        streams = parse_logfile(logfile)
        if "tel" not in streams:
            print(f"[skip] {name}: no validation loss in main.log")
            continue
        last_step = int(streams["tel"][0][-1])
        if last_step < steps:
            print(f"[skip] {name}: incomplete run (last eval step {last_step} of {steps})")
            continue
        loss = val_metric(streams["tel"][1], metric, last_k)
        if not np.isfinite(loss):
            print(f"[skip] {name}: validation loss is NaN/inf (diverged?)")
            continue

        runs.append(
            dict(
                name=name,
                bs=bs,
                lr=lr,
                wd=wd,
                steps=steps,
                tau=1.0 / (wd * lr * steps),
                loss=loss,
            )
        )
    return runs


def in_range(x, lo, hi):
    # Inclusive bounds with a small tolerance for float parsing; None means unbounded.
    if lo is not None and x < lo and not np.isclose(x, lo):
        return False
    if hi is not None and x > hi and not np.isclose(x, hi):
        return False
    return True


def group_by_bs(runs):
    groups = {}
    for r in runs:
        groups.setdefault(r["bs"], []).append(r)
    return groups


def best_per_bs(groups):
    return {bs: min(rs, key=lambda r: r["loss"]) for bs, rs in groups.items()}


# ----------------------------------------------------------------------------- plotting


def plot_sweep(groups, xkey, xlabel, ylabel, title, out_path):
    """One viridis line per batch size (smallest purple, largest yellow); red marker at each minimum."""
    batch_sizes = sorted(groups)
    colors = plt.get_cmap("viridis")(np.linspace(0, 1, len(batch_sizes)))

    fig, ax = plt.subplots(figsize=(9, 6))
    for bs, color in zip(batch_sizes, colors):
        pts = sorted(groups[bs], key=lambda r: r[xkey])
        xs = np.array([r[xkey] for r in pts])
        ys = np.array([r["loss"] for r in pts])

        ax.plot(xs, ys, "-o", color=color, markersize=7, label=f"B = {bs}")
        i = int(np.argmin(ys))
        ax.plot(
            xs[i],
            ys[i],
            "o",
            color="red",
            markersize=9,
            markeredgecolor="black",
            zorder=5,
        )

    ax.plot(
        [],
        [],
        "o",
        color="red",
        markeredgecolor="black",
        label="lowest val loss (per B)",
    )
    ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_optimal_hp(best_wd, best_lr, out_path):
    fig, ax = plt.subplots(figsize=(8, 6))
    all_bs = sorted(set(best_wd) | set(best_lr))

    if best_wd:
        bs = sorted(best_wd)
        ax.plot(
            bs,
            [best_wd[b]["wd"] for b in bs],
            "-x",
            color="tab:blue",
            markersize=10,
            markeredgewidth=2,
            label=r"Weight Decay $\lambda$",
        )
    if best_lr:
        bs = sorted(best_lr)
        ax.plot(
            bs,
            [best_lr[b]["lr"] for b in bs],
            "-x",
            color="tab:orange",
            markersize=10,
            markeredgewidth=2,
            label=r"Learning Rate $\eta$",
        )

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(all_bs)
    ax.set_xticklabels([str(b) for b in all_bs])
    ax.set_xlabel("Batch Size B")
    ax.set_ylabel("Optimal HP Value")
    ax.set_title("Optimal HP Value vs Batch Size")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def print_table(title, groups, best, xkey, xname):
    print(f"\n{title}")
    print(f"{'B':>6} {xname:>10} {'lr':>9} {'wd':>7} {'val_loss':>10}")
    print("-" * 46)
    for bs in sorted(groups):
        for r in sorted(groups[bs], key=lambda r: r[xkey]):
            mark = "  <- best" if r is best[bs] else ""
            print(
                f"{bs:>6} {r[xkey]:>10.4g} {r['lr']:>9.5f} {r['wd']:>7.4g} {r['loss']:>10.4f}{mark}"
            )
        print()


# ----------------------------------------------------------------------------- main


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--log_dir", default="my_logs")
    parser.add_argument("--out_dir", default="plots")
    parser.add_argument(
        "--wd_sweep_lr",
        type=float,
        default=0.0006,
        help="learning rate the weight decay sweep was run at",
    )
    parser.add_argument(
        "--lr_sweep_wd",
        type=float,
        default=0.1,
        help="weight decay the learning rate sweep was run at",
    )
    parser.add_argument(
        "--metric",
        choices=["min", "last"],
        default="min",
        help="min = lowest val loss during training; last = mean of last --last_k evals",
    )
    parser.add_argument("--last_k", type=int, default=5)
    parser.add_argument(
        "--lr_min", type=float, default=None, help="only plot runs with lr >= this"
    )
    parser.add_argument(
        "--lr_max", type=float, default=None, help="only plot runs with lr <= this"
    )
    parser.add_argument(
        "--wd_min", type=float, default=None, help="only plot runs with wd >= this"
    )
    parser.add_argument(
        "--wd_max", type=float, default=None, help="only plot runs with wd <= this"
    )
    parser.add_argument(
        "--model",
        choices=list(MODEL_PARAMS),
        default="mup_tiniest_1L",
        help="model size; sets TOTAL_TOKENS = 20 * params",
    )
    args = parser.parse_args()

    total_tokens = 20 * MODEL_PARAMS[args.model]

    runs = collect_runs(args.log_dir, args.metric, args.last_k, total_tokens)
    runs = [
        r
        for r in runs
        if in_range(r["lr"], args.lr_min, args.lr_max)
        and in_range(r["wd"], args.wd_min, args.wd_max)
    ]
    if not runs:
        raise SystemExit(f"No matching runs with val loss found in {args.log_dir}")
    os.makedirs(args.out_dir, exist_ok=True)

    ylabel = "Validation Loss"
    xlabel = r"AdamW timescale $\tau$"

    wd_groups = group_by_bs([r for r in runs if np.isclose(r["lr"], args.wd_sweep_lr)])
    lr_groups = group_by_bs([r for r in runs if np.isclose(r["wd"], args.lr_sweep_wd)])
    best_wd = best_per_bs(wd_groups)
    best_lr = best_per_bs(lr_groups)

    if wd_groups:
        p = os.path.join(args.out_dir, "sweeping_weight_decay.png")
        plot_sweep(
            wd_groups,
            "tau",
            xlabel,
            ylabel,
            rf"Sweeping weight decay $\lambda$ ($\eta$ = {args.wd_sweep_lr:g})",
            p,
        )
        print(f"[saved] {p}")
    else:
        print(f"[skip] weight decay plot: no runs with lr={args.wd_sweep_lr:g}")

    if lr_groups:
        p = os.path.join(args.out_dir, "sweeping_learning_rate.png")
        plot_sweep(
            lr_groups,
            "tau",
            xlabel,
            ylabel,
            rf"Sweeping learning rate $\eta$ ($\lambda$ = {args.lr_sweep_wd:g})",
            p,
        )
        print(f"[saved] {p}")
    else:
        print(f"[skip] learning rate plot: no runs with wd={args.lr_sweep_wd:g}")

    if best_wd or best_lr:
        p = os.path.join(args.out_dir, "optimal_hp.png")
        plot_optimal_hp(best_wd, best_lr, p)
        print(f"[saved] {p}")

    if wd_groups:
        print_table(
            f"Weight decay sweep (lr = {args.wd_sweep_lr:g})",
            wd_groups,
            best_wd,
            "tau",
            "tau",
        )
    if lr_groups:
        print_table(
            f"Learning rate sweep (wd = {args.lr_sweep_wd:g})",
            lr_groups,
            best_lr,
            "tau",
            "tau",
        )


if __name__ == "__main__":
    main()
