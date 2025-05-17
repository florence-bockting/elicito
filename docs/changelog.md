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
