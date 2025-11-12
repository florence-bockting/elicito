# Changelog

Versions follow [Semantic Versioning](https://semver.org/) (`<major>.<minor>.<patch>`).

Backward incompatible (breaking) changes will only be introduced in major versions
with advance notice in the **Deprecations** section of releases.

<!--
You should *NOT* be adding new changelog entries to this file,
this file is managed by towncrier.
See `changelog/README.md`.

You *may* edit previous changelogs to fix problems like typo corrections or such.
To add a new changelog entry, please see
`changelog/README.md`
and https://pip.pypa.io/en/latest/development/contributing/#news-entries,
noting that we use the `changelog` directory instead of news,
markdown instead of restructured text and use slightly different categories
from the examples given in that link.
-->

<!-- towncrier release notes start -->

## Expert prior elicitation method v0.7.0 (2025-11-12)

### 🆕 Features

- + convert eliobj.results object into xr.DataTree
  + adjust plots corresponding to new output format
  + adjust tests corresponding to new output format
  + adjust documentation to new output format
  + include xarray as new dependency ([#32](https://github.com/florence-bockting/elicito/pull/32))

### 🎉 Improvements

- + add possibility for a dry run when initializing the `Elicit` object
      + include in `Elicit` a new `meta_settings` parameter which includes a `dry_run=True` argument

  + adjust the summary information when printing `eliobj`
      + include information about shape
      + differentiate between target quantity and elicited summary
      + correct the computation of the number of hyperparameters for parametric method
      + compute number of weights (incl. biases) of NNs when using deep_prior method and include as info for computing number of hyperparameters ([#30](https://github.com/florence-bockting/elicito/pull/30))

### 🐛 Bug Fixes

- + add tests for `elicit.py` module
  + use `Enum` where appropriate to make valid options of an argument explicit. ([#30](https://github.com/florence-bockting/elicito/pull/30))
- + add tests for functions in `utils.py` module
  + discovered bug in computation of inverse_logif in `DoubleBounded` transformation ([#38](https://github.com/florence-bockting/elicito/pull/38))
- + outsource checks for `Elicit` object into new module called `_checks.py`
  + include checks when initializing the `Elicit` object and when using the `.update` method
  + add unittest for checks in `test_init.py` ([#39](https://github.com/florence-bockting/elicito/pull/39))
- + The hyperparameter names were not correctly matched with the hyperparameter values in the final output
  + This issue has been fixed in `src\elicito\_outputs.py`
  + A corresponding test has been added in `tests\unit\test_outputs.py` ([#41](https://github.com/florence-bockting/elicito/pull/41))
- + update elicito such that it is compatible with Python 3.13
  + This involved updating the dependencies in `pyproject.toml` to versions that support Python 3.13
  + relaxing scipy and pandas dependencies to allow for future versions
  + closes [Issue#19](https://github.com/florence-bockting/elicito/issues/19) ([#42](https://github.com/florence-bockting/elicito/pull/42))
- + an error occured when using the `__str__` method for the fitted eliobj
  + reason was a wrong assignment of the fitted results to the attribute (overwriting the self object)
  + fixed assignment in `fit()` method in `__init__.py`
  + added test to `tests\unit\test_init.py::test_str_method` ([#43](https://github.com/florence-bockting/elicito/pull/43))

### 📚 Improved Documentation

- + add information on print method for `eliobj` to How-To-Guides: *save-and-load* ([#25](https://github.com/florence-bockting/elicito/pull/25))


## Expert prior elicitation method v0.6.0 (2025-06-16)

No significant changes.


## Expert prior elicitation method v0.5.4 (2025-05-19)

### 🔧 Trivial/Internal Changes

- [#15](https://github.com/florence-bockting/elicito/pull/15)


## Expert prior elicitation method v0.5.3 (2025-05-18)

### 🔧 Trivial/Internal Changes

- [#14](https://github.com/florence-bockting/elicito/pull/14)


## Expert prior elicitation method v0.5.2 (2025-05-18)

### 🔧 Trivial/Internal Changes

- [#12](https://github.com/florence-bockting/elicito/pull/12), [#13](https://github.com/florence-bockting/elicito/pull/13)


## Expert prior elicitation method v0.5.1 (2025-05-17)

No significant changes.


## Expert prior elicitation method v0.5.0 (2025-05-17)

No significant changes.


## Expert prior elicitation method v0.4.0 (2025-05-17)

### ⚠️ Breaking Changes

- + integrated InvertibleNetwork implementation from BayesFlow==1.1.6 into elicito.
  + integration has been approved by BayesFlow maintainer Stefan Radev
  + removal of BayesFlow dependency enabled removing version constraints ([#11](https://github.com/florence-bockting/elicito/pull/11))


## Expert prior elicitation method v0.3.1 (2025-04-18)

### ⚠️ Breaking Changes

- Added python scripts and tests from the old package. ([#1](https://github.com/florence-bockting/elicito/pull/1))

### 🐛 Bug Fixes

- + fix check for number of model parameters. tfd.Sequential/Joint distributions were not considered in the initial check
  + add tensorflow-silence as one option to mute tensorflow warnings and other log messages ([#7](https://github.com/florence-bockting/elicito/pull/7))

### 📚 Improved Documentation

- Adjust docstrings from Sphinx layout to Mkdocs layout ([#2](https://github.com/florence-bockting/elicito/pull/2))
- + added further documentation files, tutorials
  + included option to mute progress output ([#6](https://github.com/florence-bockting/elicito/pull/6))

### 🔧 Trivial/Internal Changes

- [#8](https://github.com/florence-bockting/elicito/pull/8)


## Expert prior elicitation method v0.3.0 (2025-03-31)

No significant changes.


## Expert prior elicitation method v0.2.0 (2025-03-29)

No significant changes.


## Expert prior elicitation method v0.1.1 (2025-03-21)

No significant changes.


## Expert prior elicitation method v0.1.0 (2025-03-21)

No significant changes.


## Expert prior elicitation method v0.0.2a1 (2025-03-21)

No significant changes.
