import argparse
import ruamel.yaml as yaml

from util.config import parse_config

class ArgParser():
    def __init__(self) -> None:
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument(
            "--config",
            type=str,
            required=True, 
            help="yaml configuration file path"
        )

    def parse(self):
        args = self.parser.parse_args()

        config = yaml.load(open(args.config, "r"), Loader=yaml.Loader)
        config = parse_config(config)

        return config