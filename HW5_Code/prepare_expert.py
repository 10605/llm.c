"""Copy the complete protected seed into a NEW per-trial output directory."""
import argparse
import os
from pathlib import Path
import shutil
import struct
import tempfile


def prepare(output, seed_dir="/opt/hw5/seed"):
    output, seed_dir = Path(output), Path(seed_dir)
    if output.exists():
        raise ValueError(f"{output} already exists; choose a new trial directory. To continue it, use -y 1.")
    files = [seed_dir / name for name in (
        "model_00010919.bin", "state_00010919_00000.bin", "DONE_00010919")]
    if not all(p.is_file() for p in files):
        raise ValueError(f"{seed_dir} lacks a complete update-10919 seed checkpoint")
    with files[0].open("rb") as f:
        model = struct.unpack("<256i", f.read(1024))
    with files[1].open("rb") as f:
        state = struct.unpack("<256i", f.read(1024))
    if model[:8] != (20240326, 5, 2048, 50257, 12, 12, 768, 50304) or model[8] != 0x4D555031:
        raise ValueError("Seed model architecture/parametrization does not match the course release")
    if state[0] != 20240527 or state[10] != 10919:
        raise ValueError("Seed optimizer state does not match update 10919")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}-seed-", dir=output.parent))
    try:
        for source in files:
            shutil.copyfile(source, temporary / source.name)
        # Exclusive mkdir prevents overwriting an existing or concurrently created trial.
        output.mkdir()
        for source in files:  # completion marker is moved last
            os.rename(temporary / source.name, output / source.name)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("output")
    p.add_argument("--seed-dir", default="/opt/hw5/seed")
    args = p.parse_args()
    try:
        print("Checking and copying the protected checkpoint. The first copy can take several minutes on a new instance.",
              flush=True)
        output = prepare(args.output, args.seed_dir)
    except (ValueError, struct.error, FileExistsError) as e:
        p.error(str(e))
    print(f"Copied seed at update 10919 to {output}. Choose LR and training data; start with -y 1 -rd 1.")


if __name__ == "__main__":
    main()
