"""Run the centralized synchronous parameter-server benchmark."""

import argparse

from distributed_training.arguments import add_training_arguments, config_from_args
from distributed_training.parameter_server import launch_parameter_server


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_training_arguments(parser, distributed=True)
    args = parser.parse_args()
    launch_parameter_server(args.workers, config_from_args(args), args.out)


if __name__ == "__main__":
    main()
