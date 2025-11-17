import toml  # type: ignore [import]
import json
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--package", type=str, help="Package to modify", required=True)
args = parser.parse_args()


with open(f"packages/{args.package}/package.json", "r") as p:
    package_data = json.load(p)

with open(f"packages/{args.package}/pyproject.toml", "r") as t:
    toml_data = toml.load(t)

toml_data["project"]["version"] = package_data["version"]

with open(f"packages/{args.package}/pyproject.toml", "w") as w:
    toml.dump(toml_data, w)
