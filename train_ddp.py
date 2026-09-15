"""Run PyTorch DistributedDataParallel over the Gloo CPU backend."""

import argparse

from distributed_training.arguments import add_training_arguments, config_from_args
from distributed_training.synchronous import launch_synchronous


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_training_arguments(parser, distributed=True)
    args = parser.parse_args()
    launch_synchronous("ddp", args.workers, config_from_args(args), args.out)


if __name__ == "__main__":
    main()
