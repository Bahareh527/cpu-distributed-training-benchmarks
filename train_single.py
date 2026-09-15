"""Run the single-process baseline."""

import argparse

from distributed_training.arguments import add_training_arguments, config_from_args
from distributed_training.single import run_single


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_training_arguments(parser, distributed=False)
    args = parser.parse_args()
    run_single(config_from_args(args), args.out)


if __name__ == "__main__":
    main()
