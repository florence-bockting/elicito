# Dependency pinning and associated testing strategy

<!--
    This text comes from the copier template.
    If you find you need to update your testing strategy,
    you probably want to update this too.
-->
Here we explain our dependency pinning and associated testing strategy.
This will help you, as a user, to know what to expect
and what your options are.
As a developer, these docs can also be helpful to understand
the overall philosophy and thinking.

## Dependency pinning

We use lower-bound pinning.
In other words, we pin the lowest supported version of the packages on which we depend.
As a user, this helps you get a working install
while giving you freedom to use newer versions, should you wish.

We don't use upper-bound pins.
The reason is that we have had bad experiences with upper-bound pinning.
In the majority of cases, new releases do not cause issues
so pinning simply forces users to workaround overly strict pins[^1]
(which can be done, see
[working around incorrectly set pins][working-around-incorrectly-set-pins]).
The tradeoff with this approach is that you run the risk that,
if a dependency releases a breaking change,
the function provided by our package may break too.

[^1]:
    Yes, if the entire world followed semantic versioning perfectly,
    we could use upper-bound pins for the next major version with more confidence
    but that isn't the current state of the ecosystem.
    Even if it were, we still think this would result in unnecessary pins
    in many cases because many major releases are still compatible
    because most packages don't use the entire API of their dependencies.

### Working around incorrectly set pins

Despite our best efforts, it is possible that we will set our pins incorrectly.
Part of this is because we simply cannot test all possible combinations of package installs
(see [testing strategy][testing-strategy]),
so we might miss valid/invalid combinations.

If we set our pins incorrectly and you need to effectively overwrite them,
unfortunately there is currently no universal solution.
There has been quite some discussion,
see e.g. [this issue](https://github.com/pypa/pip/issues/8076),
but no universal resolution.

However, for some environment managers, there is a solution.
This comes in the form of dependency overrides,
which allow you to override a package's stated dependencies
(essentially fixing them on the fly,
rather than having to fix them upstream).
Here are the docs for the package managers that we know support this:

- [uv dependency overrides](https://docs.astral.sh/uv/concepts/resolution/#dependency-overrides).
- [pdm dependency overrides](https://pdm-project.org/latest/usage/dependency/#dependency-overrides).

We do not know if this strategy can be used for packaging.
For example, you are building package A.
This depends on version 2 of package B and version 1 of package C.
However, version 1 of package C (incorrectly) says
that it is only compatible with version 1 of package B.
We are not sure if the dependency overrides
can be used to release a version of package A
that can be relased to and installed from PyPI.
If this is the situation you are in and you would like a resolution,
please comment on [this issue](https://gitlab.com/openscm/copier-core-python-repository/-/issues/4).

## Testing strategy

We test against multiple python versions in our CI.
These tests run with the latest compatible versions of our dependencies
and a 'full' installation, i.e. with all optional dependencies too.
This gives us the best possible coverage of our code base
against the latest compatible version of all our possible dependencies.

In an attempt to anticipate changes to the API's of our key dependencies,
we also test against the latest unreleased version of our key dependencies once a week.
As a user, this probably won't matter too much,
except that it should reduce the chance
that a new release of one of our dependencies breaks our package
without us knowing in advance and being able to set a pin in anticipation.
As a developer, this is important to be aware of,
so we can anticipate changes as early as possible.

We additionally test with the lowest/oldest compatible versions of our direct dependencies.
This includes Python, i.e. these tests are only run
with the lowest/oldest version of Python compatible with our project.
This is because Python is itself a dependency of our project
and newer versions of Python tend to not work
with the lowest/oldest versions of our direct dependencies.
These tests ensure that our minimum supported versions are actually supported
(if they are all installed simultaneously,
see the next paragraph for why this caveat matters).
As a note for developers,
the key trick to making this work is to use `uv pip compile`
rather than `uv run` (or similar) in the CI.
The reason is that `uv pip compile`
allows you to install dependencies for a very specific combination of things,
which is different to `uv`'s normal 'all-at-once' environment handling
(for more details, see [here](https://github.com/astral-sh/uv/issues/10774#issuecomment-2601925564)).

We do not test the combinations in between lowest-supported and latest,
e.g. the oldest compatible version of package A
with the newest compatiable version of package B.
The reason for this is simply combinatorics,
it is generally not feasible
for us to test all possible combinations of our dependencies' versions.

We also don't test with the oldest versions of our dependencies' dependencies.
We don't do this because, in practice,
all that such tests actually test is
whether our dependencies have set their minimum support dependencies correctly,
which isn't our problem to solve.

Once a week, we also test what happens when a user installs from PyPI on the 'happy path'.
In other words, they do `pip install elicito`.
We check that such an install passes all the tests that don't require extras
(for developers, this is why we have `tests-min` and `tests-full` dev dependency groups,
they allow us to test a truly minimal testing environment,
separate from any extras we install to get full coverage).
Finally, we also check the installation of the locked versions of the package,
i.e. installation with `pip install 'elicito[locked]'`.
These tests give us the greatest coverage of Python versions and operating systems
and help alert us to places where users may face issues.
Having said that, these tests do require 30 separate jobs,
which is why we don't run them in CI.

Through this combination of CI testing and installation testing,
we get a pretty good coverage of the different ways in which our package can be used.
It is not perfect, largely because the combinatorics don't allow for testing everything.
If we find a particular, key, use case failing often,
then we would happily discuss whether this should be included in the CI too,
to catch issues in advance of use.
