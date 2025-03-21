<!--- --8<-- [start:description] -->
# Expert prior elicitation method

A Python package for learning prior distributions based on expert knowledge

**Key info :**
[![Docs](https://readthedocs.org/projects/elicito/badge/?version=latest)](https://elicito.readthedocs.io)
[![Main branch: supported Python versions](https://img.shields.io/python/required-version-toml?tomlFilePath=https%3A%2F%2Fraw.githubusercontent.com%2Fflorence-bockting%2Felicito%2Fmain%2Fpyproject.toml)](https://github.com/florence-bockting/elicito/blob/main/pyproject.toml)
[![Licence](https://img.shields.io/pypi/l/elicito?label=license)](https://github.com/florence-bockting/elicito/blob/main/LICENCE)

**PyPI :**
[![PyPI](https://img.shields.io/pypi/v/elicito.svg)](https://pypi.org/project/elicito/)
[![PyPI install](https://github.com/florence-bockting/elicito/actions/workflows/install-pypi.yaml/badge.svg?branch=main)](https://github.com/florence-bockting/elicito/actions/workflows/install-pypi.yaml)

**Conda :**
[![Conda](https://img.shields.io/conda/vn/conda-forge/elicito.svg)](https://anaconda.org/conda-forge/elicito)
[![Conda platforms](https://img.shields.io/conda/pn/conda-forge/elicito.svg)](https://anaconda.org/conda-forge/elicito)
[![Conda install](https://github.com/florence-bockting/elicito/actions/workflows/install-conda.yaml/badge.svg?branch=main)](https://github.com/florence-bockting/elicito/actions/workflows/install-conda.yaml)

**Tests :**
[![CI](https://github.com/florence-bockting/elicito/actions/workflows/ci.yaml/badge.svg?branch=main)](https://github.com/florence-bockting/elicito/actions/workflows/ci.yaml)
[![Coverage](https://codecov.io/gh/florence-bockting/elicito/branch/main/graph/badge.svg)](https://codecov.io/gh/florence-bockting/elicito)

**Other info :**
[![Last Commit](https://img.shields.io/github/last-commit/florence-bockting/elicito.svg)](https://github.com/florence-bockting/elicito/commits/main)
[![Contributors](https://img.shields.io/github/contributors/florence-bockting/elicito.svg)](https://github.com/florence-bockting/elicito/graphs/contributors)
## Status

<!---

We recommend having a status line in your repo
to tell anyone who stumbles on your repository where you're up to.
Some suggested options:

- prototype: the project is just starting up and the code is all prototype
- development: the project is actively being worked on
- finished: the project has achieved what it wanted
  and is no longer being worked on, we won't reply to any issues
- dormant: the project is no longer worked on
  but we might come back to it,
  if you have questions, feel free to raise an issue
- abandoned: this project is no longer worked on
  and we won't reply to any issues
-->

- prototype: the project is just starting up and the code is all prototype

<!--- --8<-- [end:description] -->

Full documentation can be found at:
[elicito.readthedocs.io](https://elicito.readthedocs.io/en/latest/).
We recommend reading the docs there because the internal documentation links
don't render correctly on GitHub's viewer.

## Installation

This package requires `tensorflow` and thus for Windows Native also the
additional requirement that Microsoft Visual C++ Redistributable for VisualStudio
are installed. See [`install tensorflow`](https://www.tensorflow.org/install/pip)

<!--- --8<-- [start:installation] -->
### As an application

If you want to use Expert prior elicitation method as an application,
then we recommend using the 'locked' version of the package.
This version pins the version of all dependencies too,
which reduces the chance of installation issues
because of breaking updates to dependencies.

The locked version of Expert prior elicitation method can be installed with

=== "mamba"
    ```sh
    mamba install -c conda-forge elicito-locked
    ```

=== "conda"
    ```sh
    conda install -c conda-forge elicito-locked
    ```

=== "pip"
    ```sh
    pip install 'elicito[locked]'
    ```

### As a library

If you want to use Expert prior elicitation method as a library,
for example you want to use it
as a dependency in another package/application that you're building,
then we recommend installing the package with the commands below.
This method provides the loosest pins possible of all dependencies.
This gives you, the package/application developer,
as much freedom as possible to set the versions of different packages.
However, the tradeoff with this freedom is that you may install
incompatible versions of Expert prior elicitation method's dependencies
(we cannot test all combinations of dependencies,
particularly ones which haven't been released yet!).
Hence, you may run into installation issues.
If you believe these are because of a problem in Expert prior elicitation method,
please [raise an issue](https://github.com/florence-bockting/elicito/issues).

The (non-locked) version of Expert prior elicitation method can be installed with

=== "mamba"
    ```sh
    mamba install -c conda-forge elicito
    ```

=== "conda"
    ```sh
    conda install -c conda-forge elicito
    ```

=== "pip"
    ```sh
    pip install elicito
    ```

Additional dependencies can be installed using

=== "mamba"
    If you are installing with mamba, we recommend
    installing the extras by hand because there is no stable
    solution yet (see [conda issue #7502](https://github.com/conda/conda/issues/7502))

=== "conda"
    If you are installing with conda, we recommend
    installing the extras by hand because there is no stable
    solution yet (see [conda issue #7502](https://github.com/conda/conda/issues/7502))

=== "pip"
    ```sh
    # To add plotting dependencies
    pip install 'elicito[plots]'

    # To add all optional dependencies
    pip install 'elicito[full]'
    ```

### For developers

For development, we rely on [uv](https://docs.astral.sh/uv/)
for all our dependency management.
To get started, you will need to make sure that uv is installed
([instructions here](https://docs.astral.sh/uv/getting-started/installation/)
(we found that the self-managed install was best,
particularly for upgrading uv later).

For all of our work, we use our `Makefile`.
You can read the instructions out and run the commands by hand if you wish,
but we generally discourage this because it can be error prone.
In order to create your environment, run `make virtual-environment`.

If there are any issues, the messages from the `Makefile` should guide you through.
If not, please raise an issue in the
[issue tracker](https://github.com/florence-bockting/elicito/issues).

For the rest of our developer docs, please see [development][development].

<!--- --8<-- [end:installation] -->

## Original template

This project was generated from this template:
[copier core python repository](https://gitlab.com/openscm/copier-core-python-repository).
[copier](https://copier.readthedocs.io/en/stable/) is used to manage and
distribute this template.
