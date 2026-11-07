"""Read CUDA/SFT logs and share the plotting implementation with vislog.ipynb."""
from pathlib import Path, PurePosixPath
import math
import re

LOG_FILES = ("main.log", "run_parameters.txt", "args.json", "model_config.json", "samples.txt")


def parse_parameters(path):
    """Support legacy pipe tables and current key=value blocks; last block wins."""
    path = Path(path)
    values = {}
    if not path.exists():
        return values
    for line in path.read_text().splitlines():
        if line.strip().startswith("|"):
            parts = [p.strip() for p in line.split("|")]
            if len(parts) >= 4:
                values[parts[1]] = parts[2]
        elif not line.startswith("command:"):
            values.update(re.findall(r"(?:^|\s)([A-Za-z_]\w*)=([^\s]+)", line))
    for output, aliases, cast in (
        ("learning_rate", ("lr", "learning rate (LR)"), float),
        ("num_parameters", ("parameters", "num_parameters"), int),
    ):
        for alias in aliases:
            if alias in values:
                try:
                    values[output] = cast(values[alias])
                    break
                except ValueError:
                    continue
    return values


def read_log(path):
    """Deduplicate resumed steps. Preserve nonfinite observations as plot gaps."""
    streams, learning_rates, times = {}, {}, {}
    for number, line in enumerate(Path(path).read_text().splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        match = re.match(r"s:(\d+)\s+(trl|tel2|tel|eval):(\S+)", line)
        if not match:
            raise ValueError(f"{path}:{number}: not a training/validation metric record")
        step, stream, value = int(match[1]), match[2], float(match[3])
        streams.setdefault(stream, {})[step] = value if math.isfinite(value) else math.nan
        if stream == "trl":
            lr = re.search(r"\blr:(\S+)", line)
            if lr:
                learning_rates[step] = float(lr[1])
            timing = re.search(r"time\s*\(ms\):(\S+)", line)
            # A later record replaces the earlier timing for a repeated step.
            times.pop(step, None)
            if timing and math.isfinite(float(timing[1])):
                times[step] = float(timing[1])
    return {
        "streams": {name: sorted(rows.items()) for name, rows in streams.items()},
        "learning_rates": sorted(learning_rates.items()),
        "training_hours": sum(times.values()) / 3_600_000,
    }


def validation_label(pattern, fallback):
    parts = PurePosixPath(pattern).parts
    if "pretraining" in parts:
        return "pretraining-mix validation"
    for part in parts:
        if re.fullmatch(r"cluster_\d+", part):
            return f"{part} validation"
    if "expert" in parts:
        return "selected-cluster validation"
    return fallback


def plot_runs(run_dirs, output=None):
    """Plot loss and actual scheduled LR; show HellaSwag only if it was logged."""
    import matplotlib.pyplot as plt

    runs = []
    for directory in run_dirs:
        directory = Path(directory)
        runs.append((directory, read_log(directory / "main.log"),
                     parse_parameters(directory / "run_parameters.txt")))
    if not runs:
        raise ValueError("Provide at least one run directory containing main.log")
    has_eval = any("eval" in data["streams"] for _, data, _ in runs)
    fig, axes = plt.subplots(1, 3 if has_eval else 2, figsize=(15 if has_eval else 11, 4.5))
    for index, (directory, data, params) in enumerate(runs):
        color = f"C{index % 10}"
        label = directory.name
        if isinstance(params.get("learning_rate"), (int, float)):
            label += f" (peak LR {params['learning_rate']:g})"
        primary = validation_label(params.get("validation", ""), "validation")
        secondary = validation_label(params.get("validation_secondary", ""), "second validation")
        for stream, style, title in (("trl", "-", "train"), ("tel", "--", primary), ("tel2", ":", secondary)):
            rows = data["streams"].get(stream, [])
            if rows:
                finite = [value for _, value in rows if math.isfinite(value)]
                minimum = f", min {min(finite):.4f}" if finite else ""
                axes[0].plot(*zip(*rows), style, color=color,
                             label=f"{label}: {title}{minimum}")
        if data["learning_rates"]:
            axes[1].plot(*zip(*data["learning_rates"]), color=color, label=label)
        if has_eval and data["streams"].get("eval"):
            axes[2].plot(*zip(*data["streams"]["eval"]), color=color, label=label)
    titles = [("Training and validation", "Loss (nats/token)"), ("Learning-rate schedule", "Actual base LR")]
    if has_eval:
        titles.append(("HellaSwag", "Accuracy"))
    for ax, (title, ylabel) in zip(axes, titles):
        ax.set(title=title, xlabel="Optimizer update", ylabel=ylabel)
        ax.grid(alpha=.25)
        if ax.lines:
            ax.legend(fontsize="small")
    fig.tight_layout()
    if output:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=180, bbox_inches="tight")
    return fig


def group_name(value):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value) or value in (".", ".."):
        raise ValueError("Experiment group must be one name containing letters, digits, _, . or -")
    return value


def upload_plan(directory, group, include_checkpoints=False):
    directory = Path(directory)
    if not directory.is_dir() or not (directory / "main.log").is_file():
        raise ValueError(f"{directory} must be a run directory containing main.log")
    group_name(directory.name)
    prefix = f"experiments/{group_name(group)}/{directory.name}"
    paths = [directory / name for name in LOG_FILES if (directory / name).is_file()]
    if include_checkpoints:
        for pattern in ("model_*.bin", "state_*.bin", "DONE_*"):
            paths.extend(sorted(directory.glob(pattern)))
    return [(path, f"{prefix}/{path.name}") for path in paths]


def upload_run(client, bucket, directory, group, include_checkpoints=False):
    plan = upload_plan(directory, group, include_checkpoints)
    for path, key in plan:
        client.upload_file(str(path), bucket, key)
    return [key for _, key in plan]


def download_logs(client, bucket, group, destination):
    """Download logs only, even if the same S3 prefix also contains checkpoints."""
    prefix = f"experiments/{group_name(group)}/"
    destination = Path(destination)
    saved = []
    for page in client.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
        for item in page.get("Contents", []):
            key = item["Key"]
            if not key.startswith(prefix):
                continue
            relative = PurePosixPath(key[len(prefix):])
            if len(relative.parts) != 2 or relative.name not in LOG_FILES or any(p in (".", "..") for p in relative.parts):
                continue
            path = destination / group / relative.parts[0] / relative.name
            path.parent.mkdir(parents=True, exist_ok=True)
            client.download_file(bucket, key, str(path))
            saved.append(path)
    return saved
