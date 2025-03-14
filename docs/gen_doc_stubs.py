"""
Generate virtual doc files for the mkdocs site.

This script can also be run directly to actually write out those files,
as a preview.

Credit to the creators of:
https://oprypin.github.io/mkdocs-gen-files/
and the docs at:
https://mkdocstrings.github.io/crystal/quickstart/migrate.html
and the mainters of:
https://github.com/mkdocstrings/python
"""

from __future__ import annotations

import importlib
import pkgutil
from pathlib import Path

import mkdocs_gen_files
from attrs import define

ROOT_DIR = Path("api")
PACKAGE_NAME_ROOT = "elicito"
nav = mkdocs_gen_files.Nav()


@define
class PackageInfo:
    """
    Package information used to help us auto-generate the docs

    Not stricly needed anymore now that mkdocstrings-python has a summary option,
    but being kept in case we need something like this pattern again.
    """

    full_name: str
    stem: str
    summary: str


def write_subpackage_pages(subpackage: object) -> tuple[PackageInfo, ...]:
    """
    Write pages for the sub-packages of a module
    """
    sub_sub_packages = []
    for _, name, is_pkg in pkgutil.walk_packages(subpackage.__path__):
        subpackage_full_name = subpackage.__name__ + "." + name
        sub_package_info = write_package_page(subpackage_full_name)
        sub_sub_packages.append(sub_package_info)

    return tuple(sub_sub_packages)


def get_write_file(package_full_name: str) -> Path:
    """Get directory in which to write the doc file"""
    write_dir = ROOT_DIR
    for sub_dir in package_full_name.split(".")[:-1]:
        write_dir = write_dir / sub_dir

    write_file = write_dir / package_full_name.split(".")[-1] / "index.md"

    return write_file


def write_package_page(
    package_full_name: str,
) -> PackageInfo:
    """
    Write the docs pages for a package (or sub-package)
    """
    package = importlib.import_module(package_full_name)

    if hasattr(package, "__path__"):
        write_subpackage_pages(package)

    package_name = package_full_name.split(".")[-1]

    write_file = get_write_file(package_full_name)

    nav[package_full_name.split(".")] = write_file.relative_to(
        ROOT_DIR / PACKAGE_NAME_ROOT
    ).as_posix()

    with mkdocs_gen_files.open(write_file, "w") as fh:
        fh.write(f"# {package_full_name}\n")

        fh.write("\n")
        fh.write(f"::: {package_full_name}")

    package_doc_split = package.__doc__.splitlines()
    if not package_doc_split[0]:
        summary = package_doc_split[1]
    else:
        summary = package_doc_split[0]

    return PackageInfo(package_full_name, package_name, summary)


write_package_page(PACKAGE_NAME_ROOT)
with mkdocs_gen_files.open(ROOT_DIR / PACKAGE_NAME_ROOT / "NAVIGATION.md", "w") as fh:
    fh.writelines(nav.build_literate_nav())
