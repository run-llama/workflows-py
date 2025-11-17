import toml  # type: ignore [import]
import requests
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--package", type=str, help="Package to modify", required=True)
args = parser.parse_args()


def get_latest_version(package_name: str) -> str:
    url = f"https://pypi.org/pypi/{package_name}/json"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return data["info"]["version"]
    else:
        raise Exception(f"Package '{package_name}' not found on PyPI.")


with open("pyproject.toml", "r") as t:
    toml_data = toml.load(t)

if toml_data["project"]["version"] == get_latest_version(args.package):
    print("1")
else:
    print("0")
