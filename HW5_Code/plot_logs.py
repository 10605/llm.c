"""Plot one or more CUDA/SFT output directories without AWS credentials."""
import argparse
from experiment_logs import plot_runs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("runs", nargs="+", help="output directories containing main.log")
    p.add_argument("--output", default="plots/training.png")
    args = p.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    plot_runs(args.runs, args.output)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
