"""Create learning, timing, and speedup plots from benchmark CSV files."""

import argparse

from distributed_training.plotting import create_plots


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logs", default="logs")
    parser.add_argument("--output", default="artifacts/figures")
    args = parser.parse_args()
    for path in create_plots(args.logs, args.output):
        print(f"saved {path}")


if __name__ == "__main__":
    main()
