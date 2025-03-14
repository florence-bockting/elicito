"""
Write out the pins for our conda recipe

These can be copy-pasted into the locked info at
https://github.com/conda-forge/elicito-feedstock/blob/main/recipe/meta.yaml
"""

from __future__ import annotations

import tomli
import typer
from attrs import define
from packaging.version import Version


@define
class VersionInfoHere:
    """Version info class for use in this script"""

    name: str
    min_pin: str
    max_pin: str


def main() -> None:
    """
    Write our docs environment file for Read the Docs
    """
    PYPROJECT_TOML_FILE = "pyproject.toml"
    REQUIREMENTS_LOCK_FILE = "requirements-locked.txt"

    with open(PYPROJECT_TOML_FILE, "rb") as fh:
        pyproject_toml = tomli.load(fh)

    with open(REQUIREMENTS_LOCK_FILE) as fh:
        requirements_info = fh.read().splitlines()

    pypi_to_conda_name_map = {
        "scitools-iris": "iris",
        "cf-xarray": "cf_xarray",
        "typing-extensions": "typing_extensions",
    }
    version_info_l = []
    for dependency in pyproject_toml["project"]["dependencies"]:
        package_name = (
            dependency.split(">")[0].split("<")[0].split(">=")[0].split("<=")[0]
        )

        for line in requirements_info:
            if line.startswith(package_name):
                package_version_line = line
                break
        else:
            msg = f"Didn't find pin information for {package_name}"
            raise AssertionError(msg)

        version = package_version_line.split("==")[-1]
        vv = Version(version)
        max_pin = f"{vv.major}.{vv.minor}.{vv.micro + 1}"

        if package_name in pypi_to_conda_name_map:
            conda_name = pypi_to_conda_name_map[package_name]
        else:
            conda_name = package_name

        version_info_l.append(
            VersionInfoHere(name=conda_name, min_pin=version, max_pin=max_pin)
        )

    print("Pins for library")
    for vi in version_info_l:
        print(f"- {vi.name}")

    print("")
    print("Pins for application")
    for vi in version_info_l:
        print(
            f"- {{{{ pin_compatible('{vi.name}', lower_bound='{vi.min_pin}', upper_bound='{vi.max_pin}') }}}}"  # noqa: E501
        )


if __name__ == "__main__":
    typer.run(main)
