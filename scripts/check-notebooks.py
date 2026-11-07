"""
Check that every executed notebook of the docs is up to date

The docs render the committed notebooks and execute nothing. For each
jupytext page (a `.py` file with a jupytext header) the check fails if

1. the paired `.ipynb` is missing,
2. a cell of the notebook differs from the `.py` page in type, source or tags,
3. the code cells did not run once from top to bottom, in one kernel.

It also fails for a notebook without a page, because the docs would render it.

Run `make docs-notebooks` to update the notebooks.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import jupytext


def is_page(path: Path) -> bool:
    """Return whether a `.py` file is a jupytext page"""
    if ".ipynb_checkpoints" in path.parts:
        return False
    with open(path) as fh:
        head = fh.read(400)
    return "jupytext:" in head


def cells(notebook: Any) -> list[tuple[str, str, list[str]]]:
    """Return the type, the source and the tags of each cell"""
    return [
        (cell.cell_type, cell.source.strip(), sorted(cell.metadata.get("tags", [])))
        for cell in notebook.cells
    ]


def check(page: Path) -> list[str]:
    """Return the problems of the notebook that belongs to one page"""
    path = page.with_suffix(".ipynb")
    if not path.exists():
        return [f"{path}: missing"]

    problems = []
    notebook = jupytext.read(path)
    if cells(jupytext.read(page)) != cells(notebook):
        problems.append(f"{path}: the cells differ from {page.name}")

    counts = [
        cell.get("execution_count")
        for cell in notebook.cells
        if cell.cell_type == "code"
    ]
    if counts != list(range(1, len(counts) + 1)):
        problems.append(f"{path}: the code cells did not run once from top to bottom")
    return problems


def main() -> int:
    """Check the notebooks and print each problem"""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        default=[Path("docs")],
        help="pages or folders to check (default: docs)",
    )
    args = parser.parse_args()

    pages = sorted(
        page
        for path in args.paths
        for page in (path.rglob("*.py") if path.is_dir() else [path])
        if is_page(page)
    )
    problems = [problem for page in pages for problem in check(page)]
    problems += [
        f"{notebook}: no page {notebook.with_suffix('.py').name}"
        for path in args.paths
        if path.is_dir()
        for notebook in sorted(path.rglob("*.ipynb"))
        if ".ipynb_checkpoints" not in notebook.parts
        and not notebook.with_suffix(".py").exists()
    ]
    for problem in problems:
        print(problem)

    if problems:
        print(
            f"\n{len(problems)} problems in {len(pages)} pages.",
            "Run `make docs-notebooks`.",
        )
        return 1
    print(f"All {len(pages)} notebooks are up to date.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
