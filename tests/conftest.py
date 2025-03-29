"""
Re-useable fixtures etc. for tests

See https://docs.pytest.org/en/7.1.x/reference/fixtures.html#conftest-py-sharing-fixtures-across-multiple-files
"""

import pytest


@pytest.fixture(scope="session", autouse=True)
def pandas_terminal_width():
    pandas = pytest.importorskip("pandas")

    # Set pandas terminal width so that doctests don't depend on terminal width.

    # We set the display width to 120 because examples should be short,
    # anything more than this is too wide to read in the source.
    pandas.set_option("display.width", 120)

    # Display as many columns as you want (i.e. let the display width do the
    # truncation)
    pandas.set_option("display.max_columns", 1000)
