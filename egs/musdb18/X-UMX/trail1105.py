import yaml
import os
import argparse

parser = argparse.ArgumentParser()

with open("local/conf.yml") as f:
    def_conf = yaml.safe_load(f)


print(def_conf.keys())
from asteroid.utils import prepare_parser_from_dict, parse_args_as_dict
parser = prepare_parser_from_dict(def_conf, parser=parser)