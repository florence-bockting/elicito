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

## Expert prior elicitation method v0.8.0 (2026-11-08)

### ⚠️ Breaking Changes

- + `Elicit.fit()` no longer prompts when the object is already fitted. It raises
    `ValueError`. Pass `overwrite=True` to fit again and replace the results.
  + `Elicit.save()` no longer prompts when the target file exists. It raises
    `FileExistsError`. Pass `overwrite=True` to replace the file.
      + a script, a notebook, a CI job and a `joblib` worker no longer hang on
        the removed `input()` prompts
  + `elicito` no longer silences warnings for the whole process at import time.
    Warnings from every library in your session are visible again.
  + the elicitation workflow no longer writes `seed_chain` into the `trainer`
    dictionary you passed in. Read the seed from `results["seed"]` instead. ([#45](https://github.com/florence-bockting/elicito/pull/45))
- + `eliobj.results` no longer holds the groups `prior`, `model`,
    `target_quantity` and `elicited_summary`. Every one of them is a function
    of the learned hyperparameters, which the new group `learned_weights`
    stores. Call `eliobj.sample()` to compute them, and choose the number of
    draws yourself. The saved file is much smaller: the results of a test
    model with three parameters and five epochs went from 2.0 MB to 164 kB.
  + `el.plots.prior_joint()`, `prior_marginals()`, `elicits()`,
    `priorpredictive()` and `prior_averaging()` accept the samples through the
    new `samples` argument. Without it, each plot runs one forward simulation
    of its own.
  + `print(eliobj)` reports the names of the target quantities and elicited
    summaries of a fitted object, without their shapes. The shapes follow from
    `B` and `num_samples`. ([#54](https://github.com/florence-bockting/elicito/pull/54))
- + `el.utils.load()` is removed. Load a saved object with `el.Elicit.load(file)`,
    the counterpart of `eliobj.save()`. ([#59](https://github.com/florence-bockting/elicito/pull/59))
- + The modules are grouped by building block: `el.parameters`, `el.initializers`
    and `el.optimizers` are subpackages, and `el.models` holds the generative
    model. The top-level names `el.Elicit`, `el.model()`, `el.parameter()`,
    `el.target()`, `el.expert`, `el.optimizer()`, `el.trainer()` and
    `el.initializer()` are unchanged. These paths moved:

    | Before | After |
    | :-- | :-- |
    | `el.initialization.uniform` | `el.initializers.uniform` |
    | `el.initialization.hyper_names` | `el.optimizers.search.hyper_names` |
    | `el.utils.LowerBound`, `UpperBound`, `DoubleBound` | `el.parameters.LowerBound`, `UpperBound`, `DoubleBound` |
    | `el.warmstart.score`, `compile_score`, `evaluate`, `PENALTY` | `el.optimizers.search` |
    | `el.warmstart.warm_start` | `el.initializers.warmstart.warm_start` |
    | `el.cmaes` | `el.optimizers.cmaes` |
    | `el.optimization` | `el.optimizers.sgd` |
    | `el.simulations.Priors`, `intialize_priors`, `sample_from_priors` | `el.parameters.priors` |
    | `el.simulations` (the forward simulation) | `el.models` |
    | `el.methods` | `el.parameters.methods` |
    | `el.elicit` (the constructors) | `el.specs` |
    | `elicito.networks` (`el.networks` still works) | `elicito.parameters.networks` |
    | `el.specs.initializer` (`el.initializer` still works) | `el.initializers.initializer` |
    | `el.utils.dry_run` | `el.elicit.dry_run` |
    | `el.utils.save`, `save_as_pkl`, `read_storage` | removed; use `eliobj.save()` and `el.Elicit.load()` |

    Use the table to update your imports. An eliobj with a `network` that was
    saved with an older version cannot be loaded. ([#60](https://github.com/florence-bockting/elicito/pull/60))
- + `el.target()` is removed. Use one constructor for each elicitation technique:
    `el.quantiles(name, probs=...)`, `el.correlation(name)`, `el.identity(name)`
    and `el.custom(name, func, loss)`. The loss defaults to `MMD2(kernel="energy")`
    for `el.quantiles` and `el.identity`, and to `L2` for `el.correlation`. The
    weight defaults to 1.0. The argument `target_method=` is now `output=`.
  + The expert answers are keyed by the target name, without the prefix of the
    technique: `"y_X0"`, not `"quantiles_y_X0"`. `el.utils.get_expert_datformat()`
    is removed. An answer with a missing target, or with the wrong number of
    quantiles, raises a `ValueError` with the target name. The target names must
    be unique, and `prior`, `total_loss` and `penalty` are reserved.
  + The table "Loss by target" in `print(fit)` shows the elicitation technique
    of each target. ([#64](https://github.com/florence-bockting/elicito/pull/64))
- + `Elicit` takes flat settings: `el.Elicit(model, targets, parameters, *,
    network=None, initializer=None, optimizer="auto", cmaes=None, seeds=(0,),
    epochs=600, B=128, num_samples=200, kappa=0.0, verbose=True, dry_run=True)`.
    `targets` now comes before `parameters`. `el.trainer()`, `el.optimizer()`,
    `el.meta_settings()` and `el.utils.parallel()` are removed.
  + The method is not an argument. A `network` means a deep prior.
  + `optimizer="auto"` uses CMA-ES for parametric priors and Adam for a deep
    prior. `optimizer="adam"` uses a learning rate of 0.1. For other settings,
    give a `tf.keras` optimizer, for example
    `tf.keras.optimizers.Adam(learning_rate=0.01)`. The CMA-ES settings go to
    `cmaes=dict(sigma0=..., popsize=...)`. The default is `epochs=600`.
  + `fit(parallel=...)` is removed. One replication runs for each of the
    `seeds`, in parallel. Seeds are no longer drawn at random.
  + The expert answers go to `fit(experts)`, not to `Elicit(expert=...)`. Give
    them as a dict keyed by target name. `el.expert.data()` is removed.
    `el.expert.simulator(...)` also goes to `fit()`.
  + `update()` takes the arguments of `Elicit`, for example `update(epochs=200)`.
  + `cma` is a core dependency. The extra `elicito[cma]` is removed.
  + An eliobj saved with an earlier version cannot be loaded. ([#65](https://github.com/florence-bockting/elicito/pull/65))
- + `el.model()` is removed. Give `Elicit` the model as a callable, for example
    an instance of the model class. Bind other model arguments with
    `functools.partial`, for example
    `el.Elicit(partial(ToyModel(), design_matrix=X), targets, parameters)`.
  + `Elicit` accepts a PyMC model. elicito reads its priors, so `parameters` is
    optional. A given list replaces the priors that elicito reads.
  + `el.quantiles`, `el.identity` and `el.custom` take `select=`. An integer
    indexes the last axis of the output. A dict maps the dims of a PyMC output
    to coord labels, for example `select={"participant": 3, "delay": 12}`.
    `select=` replaces a closure such as `output=lambda ypred: ypred[:, :, 3]`. ([#66](https://github.com/florence-bockting/elicito/pull/66))
- + `Elicit.load` is removed. Use `el.load(file)`. It returns the `Fit` of a
    fitted `eliobj`, the `Panel` of a fitted panel, else the `Elicit` object.
    `fit.eliobj` gives the `Elicit` object of a loaded fit, for example to fit
    it again.
  + A fitted panel of experts refits only with `fit(experts, overwrite=True)`,
    as one expert does. ([#69](https://github.com/florence-bockting/elicito/pull/69))

### 🆕 Features

- Added `from_elicits`, which derives the
  initialization box from the expert-elicited statistics during `fit`. Pass it to
  `el.initializer(distribution=...)` instead of `uniform` to avoid guessing a
  `mean` and a `radius` that match the scale of the hyperparameters.

  Added `el.initializer(method="warmstart")`. It runs a derivative-free
  Nelder-Mead search on the unconstrained hyperparameters before training, using
  the same loss on fewer prior draws. The search needs no gradient, so it cannot
  diverge. It costs one forward simulation per evaluation. It is meant for a box
  you do not trust: it rescues a badly placed box, and does not improve a
  well-placed one.

  [`initializer`][elicito.initializers.spec.initializer] now defaults to a Nelder-Mead warm
  start on a box derived from the expert data, so `el.initializer()` needs no
  argument. `distribution` and `iterations` are optional for every method; an
  omitted `iterations` takes the default of the method, 100 objective
  evaluations for "warmstart" and 32 candidates for the three samplers. ([#46](https://github.com/florence-bockting/elicito/pull/46))
- Select a subset of the model parameters in `el.plots.prior_marginals()` and `el.plots.prior_joint()` with the new `params` argument. ([#51](https://github.com/florence-bockting/elicito/pull/51))
- Add `kappa` to `el.trainer()`. It weights a penalty on the negative mean log standard deviation of the priors, which keeps a non-identified prior away from a point mass.

  The penalty value is recorded in `history_stats.loss.penalty` at every epoch, also at `kappa=0`. ([#53](https://github.com/florence-bockting/elicito/pull/53))
- `Elicit.sample()` simulates from the learned prior.

  The results of a fitted `eliobj` now hold the learned trainable variables in
  the group `learned_weights`. Together with the generative model and the
  parameter definitions, they reproduce the prior samples, the model
  simulations, the target quantities and the elicited summaries. `sample()`
  returns those four groups as an `xarray.DataTree`, one forward pass per
  replication.

  ```python
  eliobj.fit()
  samples = eliobj.sample(num_samples=1_000)
  el.plots.prior_marginals(eliobj, samples=samples)
  ```

  `num_samples`, `B` and `seed` are free. With the training values, and the
  seed of the replication, the draws differ from the last training epoch by
  one optimizer step: the epoch samples before the update, and the stored
  variables are the ones after it. ([#54](https://github.com/florence-bockting/elicito/pull/54))
- Add `el.utils.add_derived()`, which stores a derived parameter in the prior samples of a fitted `eliobj`. The plotting functions can then select it by name. ([#55](https://github.com/florence-bockting/elicito/pull/55))
- Added `el.initializer(method="cmaes")`, a global search for the start values
  with CMA-ES. It reads the whole initialization box, not only its centre, and
  keeps every candidate inside it. Use it when you cannot place the box around
  the answer: the samplers and the Nelder-Mead warm start take the first basin
  they meet, which need not be the best one. It needs the optional dependency
  `cma`, and its default budget is 500 objective evaluations. ([#56](https://github.com/florence-bockting/elicito/pull/56))
- Added CMA-ES as an alternative to the gradient descent for `method="parametric_prior"`.
  Select it with `el.optimizer(optimizer="cmaes", sigma0=...)`.
  The search needs no gradient, so it cannot diverge, and it can leave a local basin.
  `trainer(epochs=...)` is then the budget in forward simulations.
  `sigma0` also takes a dictionary `{hyperparameter name: step size}`, for a model
  whose hyperparameters do not live on the same scale.
  With `el.initializer(method="cmaes")`, the initialization runs no search of its
  own. It only reads the box, which gives the training its start point and its
  first step size per hyperparameter. `iterations` is then not used. ([#56](https://github.com/florence-bockting/elicito/pull/56))
- Added `Elicit.hyperparameters(replication=0)`. It returns the learned
  hyperparameter values of a fitted `eliobj` with `method="parametric_prior"`,
  on the scale of the prior family. The values belong to the point that the fit
  returns. For CMA-ES, this is the best point of the run, which can differ from
  the last recorded generation. ([#62](https://github.com/florence-bockting/elicito/pull/62))
- Added the module `elicito.adapter.pymc`. It connects elicito to a PyMC model
  for `method="parametric_prior"`. `parameters(model)` reads the priors,
  `model(model)` translates the PyMC model into a generative model, and
  `set_hyperparameters(model, eliobj)` writes the learned values back with
  `pm.set_data`. Each hyperparameter must be a named `pm.Data` node. A prior can
  be Normal, HalfNormal, Uniform, Beta or Gamma. The generative model samples
  every other free random variable, also a multivariate one such as Dirichlet,
  MvNormal, Wishart or LKJCholeskyCov. The adapter needs Python 3.12 or later
  and the optional dependency `pymc`: `pip install "elicito[pymc]"`. ([#62](https://github.com/florence-bockting/elicito/pull/62))
- `eliobj.fit()` now returns a `Fit` object with the learned priors.
  `fit.best()` gives the replication with the lowest loss, and
  `fit.hyperparameters()` the learned values of that replication.
  `fit.sample()` caches its default call. `fit.plot.<name>()` calls the plot
  functions of `elicito.plots` with the cached samples. `fit.summary()` returns
  the hyperparameters, the priors, the fit to the expert and the loss of each
  target as an `xr.DataTree`, for the best replication, one replication or all
  of them. `print(fit)` shows this summary as text tables. ([#63](https://github.com/florence-bockting/elicito/pull/63))
- + `Elicit` takes `pool=` for a panel of experts. Give `fit()` a dict of
    answers keyed by expert name, for example
    `fit({"expert 1": {...}, "expert 2": {...}})`. It returns a `Panel`.
    `pool=None` fits each expert alone. `pool="complete"` fits one prior for
    all experts. `pool=el.partial(over=["mu0"])` gives each expert an offset
    from the panel value of `mu0`, and estimates the spread `tau` in two
    stages. A fixed `tau=` skips stage 1.
  + `Panel.population_prior()` samples the prior of the whole panel. ([#67](https://github.com/florence-bockting/elicito/pull/67))
- `save()` keeps a fitted panel of experts: the fit of each expert, or the
  joint fit and `tau`. `el.load()` returns the `Panel` again. ([#69](https://github.com/florence-bockting/elicito/pull/69))

### 🎉 Improvements

- + introduce a strategy object per prior-learning method in the new
    `elicito.methods` module
      + `parametric_prior` and `deep_prior` each provide their own `build`,
        `sample`, `trainable_variables`, `new_history`, `record_epoch`,
        `finalize`, `check`, `initialize` and `init_matrix_slice`
      + `_checks.py`, `initialization.py`, `optimization.py`, `simulations.py`
        and `utils.py` no longer branch on `trainer["method"]`
  + move `PriorMethods`, `ProgressMethod`, `SamplingMethod` and `VariableType`
    from `elicit.py` into `types.py` to break an import cycle
      + the names stay importable from `elicito.elicit`
  + an unknown value for `trainer["method"]` now raises a clear `ValueError`
    that lists the valid methods ([#44](https://github.com/florence-bockting/elicito/pull/44))
- + the ground truth is sampled once per check instead of once per target
  + the base distribution of the `deep_prior` method is built once, not on every
    epoch
  + the constraint map of the `parametric_prior` method is built once per
    training run ([#45](https://github.com/florence-bockting/elicito/pull/45))
- `from_elicits` now derives the box from
  the role of a hyperparameter, not from its constraint alone. A shape, such as
  the Weibull concentration, has no relation to the scale of the elicited data,
  so its box covers the natural range 1 to 5. Any other lower-bounded
  hyperparameter spans from `spread / 100` up to the pooled 95% quantile, which
  covers a small prior scale and a data scale in one box.

  The initialization methods now share one interface. `InitMethod` in
  `elicito.initialization` declares `check`, `propose` and `dry_run_slice`, and a
  registry maps a method name to its class. Adding a method no longer means an
  edit in five places. The valid names in the error message of
  [`initializer`][elicito.initializers.spec.initializer] now come from that registry. ([#46](https://github.com/florence-bockting/elicito/pull/46))
- Update the dependencies, and keep TensorFlow 2.21 and NumPy 2.5 working.

  TensorFlow moves from 2.20 to 2.21, and NumPy from 2.1 to 2.5.
  `tensorflow-probability` stays at 0.25.0, its newest release.

  NumPy 2.5 rejects `float()` on an array that is not 0-dimensional. The warm
  start read the loss that way, so it now reduces the shape first.

  The new stubs of `types-tensorflow` and `matplotlib` 3.11 report the shape
  arguments in `networks.py` and the `subfigures` result in `plots.py`
  differently. Both are now typed for what they hold at runtime. ([#47](https://github.com/florence-bockting/elicito/pull/47))
- Remove the `pandas` dependency, and with it the `elicito[pandas]` extra.

  `load` read the file with `pandas.read_pickle`, although `save` writes it with
  `pickle.dump`. It now reads the file with `pickle.load`. The model averaging
  plot built a two-column data frame only to sort the seeds by weight, which
  `numpy.argsort` does.

  `el.plots.prior_averaging` works again without parallelization. The seed was
  passed as a string, so the data frame read it one character at a time and the
  call ended in a length error. ([#48](https://github.com/florence-bockting/elicito/pull/48))
- Compile the training with `tf.function`, and seed every draw statelessly.

  The forward simulation, the loss and the gradient of one epoch are now traced
  once and re-used. Eager execution dispatched every operation of the epoch from
  Python, and the tensors of one epoch are small, so that dispatch decided the
  runtime. Measured on the human growth model, one simulation with its loss went
  from 184 ms to 4 ms, and one Adam epoch from 424 ms to 15 ms.

  A compiled graph cannot use `tf.random.set_seed`: it clears the kernel caches,
  and the next call rebuilds the whole graph. So every distribution is now seeded
  with a stateless seed pair instead. A generative model that names a `seed`
  argument is given one, and its draws then repeat as well.

  The search of `warm_start` keeps one prior model and writes each candidate
  into its variables, instead of building a prior model per evaluation.

  AutoGraph cannot convert `sample_from_priors` on Python 3.13: it raises an
  `AssertionError` while it builds the control-flow graph. TensorFlow 2.21 does
  not fix this. The function needs no conversion, because it branches on Python
  values only, so it now carries `tf.autograph.experimental.do_not_convert`.
  This removes the warning that every `fit()` call printed. ([#52](https://github.com/florence-bockting/elicito/pull/52))
- With `optimizer="cmaes"`, the `initializer` is now optional.
  Without it, `Elicit` uses `initializer(method="cmaes")`.
  This default reads the start point and the first step size
  from the scale of the expert data.
  `Elicit.update(initializer=None)` applies the same default.
  `SamplingMethod` has the new member `cmaes`. ([#57](https://github.com/florence-bockting/elicito/pull/57))
- Show the progress of the training and the initialization as a table.

  The row shows the current loss, the number of skipped steps,
  the speed, the elapsed time and the remaining time.
  CMA-ES and the initialization show the loss only.
  A parallel fit shows one row for each seed.
  The table uses `rich`, so the `tqdm` dependency is removed. ([#58](https://github.com/florence-bockting/elicito/pull/58))

### 🐛 Bug Fixes

- + the default optimizer is usable again. It was stored as an optimizer
    instance, so a second run reused the state of the first one.
  + training now stops before a NAN gradient reaches the trainable variables, so the last valid values are kept.
  + a custom target function now receives only the arguments it declares.
  + initialization results are labelled from the initialization matrix, so every hyperparameter name matches its value.
  + a Pearson correlation of exactly `0.0` is no longer dropped. The correlation target failed with a reshape error whenever it occurred.
  + two `Elicit` objects no longer share one `meta_settings` dictionary.
  + eight error messages were raised as tuples and printed as `("...",)`. They
    are plain strings now.
  + `initializer(method="random")` no longer needs `scipy`. The `scipy` import is now made only for the `"sobol"` and `"lhs"` methods. ([#45](https://github.com/florence-bockting/elicito/pull/45))
- Fixed three ways the initialization could select a start value that cannot
  train.

  + A candidate whose forward simulation overflows is now rejected. A quantile
    query hides an overflow, because the 95% quantile of a sample with a few
    infinite draws is still finite. The target quantities are checked instead of
    the loss alone.
  + The `warmstart` search no longer returns a point that failed. It keeps the
    best usable point, and it says so when every evaluated point fails.
  + The `warmstart` search restarts its simplex until `iterations` is spent.
    Nelder-Mead converged on its own tolerances after about 110 evaluations,
    whatever budget was given.
  + The candidate design of `"sobol"` and `"lhs"` filled only a diagonal of the
    box when `mean` and `radius` were given per hyperparameter, as
    `from_elicits` does. One one-dimensional sequence was drawn per
    hyperparameter, and the successive blocks run in nearly the same order, so
    the columns correlated above 0.97. One design over all hyperparameters is
    drawn now. ([#46](https://github.com/florence-bockting/elicito/pull/46))
- Two defects of the `deep_prior` method are fixed.

  `B` and `num_samples` could not be changed. The coupling networks were built
  again on every call, and Keras stores the batch dimensions of the first
  build. A fit with `B=64` raised `Sequential model has already been configured
  to use input shape (128, 200, 2)`. The network is now built once.

  A normalizing flow could not be rebuilt from its seed. The permutation layers
  draw from the global generator of `numpy`, which `tf.random.set_seed` does
  not control, so two networks built from the same seed permuted differently.
  The construction now runs with a seeded generator, and restores the state of
  the caller. ([#49](https://github.com/florence-bockting/elicito/pull/49))
- `el.plots.prior_joint` no longer raises `OverflowError` for a collapsed
  parameter.

  A kernel density needs at least three distinct values. A parameter whose draws
  take only two, such as a truncated prior that has collapsed onto its bounds,
  made the bandwidth infinite inside `arviz_stats`. Such a panel now shows one
  line per value, and a warning names the parameter.

  A density that fails for any other reason now also leaves its panel empty, with
  a warning that reports the range and the standard deviation of the draws. One
  parameter no longer costs the whole figure. The draws are passed as float64,
  so a wide sample no longer overflows the variance.

  Two other defects of the same plot are fixed. The off-diagonal panels were
  transposed, so a column did not share its x-axis with the density on its
  diagonal. A model with one parameter could not be plotted at all. ([#50](https://github.com/florence-bockting/elicito/pull/50))
- `el.plots.hyperparameter` works again after a fit whose initializer draws no
  candidates.

  It read the hyperparameter names from `results.initialization`, and
  `el.initializer(method="cmaes")` writes no such group. The call ended in
  `AttributeError: 'DataTree' object has no attribute 'initialization'`. The
  names now come from the model, in the order the initializer uses.

  `el.plots.initialization` needs that group, and says so instead of raising the
  same `AttributeError`. ([#56](https://github.com/florence-bockting/elicito/pull/56))
- `el.expert.simulator(ground_truth=...)` gives each prior to the parameter that
  its key names, when the keys are the parameter names. Before, it ignored the
  keys and used the order of the dict, so a prior could go to the wrong parameter. ([#62](https://github.com/florence-bockting/elicito/pull/62))
- `fit()` accepts a Keras optimizer with a learning-rate schedule, for example
  `tf.keras.optimizers.Adam(learning_rate=ExponentialDecay(...))`. Before,
  the rebuild of the optimizer failed. ([#68](https://github.com/florence-bockting/elicito/pull/68))
- A parallel fit in a notebook uses the right model. A worker got a model from a
  notebook cell without its source, so TensorFlow autograph read the source from
  a wrong file. It converted a different function, and the fit failed, for
  example with `got an unexpected keyword argument`. ([#68](https://github.com/florence-bockting/elicito/pull/68))
- `save(file=...)` accepts an absolute path. Before, it wrote the file below
  the working directory. ([#69](https://github.com/florence-bockting/elicito/pull/69))

### 📚 Improved Documentation

- Added the how-to guide *Choose an initialization method*. It shows exact
  values, a user-set `uniform` box, [`from_elicits`]
  [elicito.initialization.from_elicits] and the Nelder-Mead warm start on the
  loss surface of a two-hyperparameter model, and then on a Weibull model where
  a bad start value makes the draws overflow. ([#46](https://github.com/florence-bockting/elicito/pull/46))
- Added the developer page *Package architecture*. It shows the workflow of the
  `Elicit` object, the import layers of the subpackages, a dependency structure
  matrix of all modules, the cohesion and instability of each subpackage, and a
  check of the Stable Dependencies Principle. ([#60](https://github.com/florence-bockting/elicito/pull/60))
- Added the how-to guide "Use a PyMC model". The docs now build on Python 3.12,
  because the guide needs `pymc`. ([#62](https://github.com/florence-bockting/elicito/pull/62))
- Added nine case studies with models of the book *Bayesian Cognitive Modeling*
  (Lee & Wagenmakers, 2013), ported from pymc-resources. Each case study
  translates one PyMC model with `elicito.adapter.pymc` and checks the TensorFlow
  samples against `pm.draw`. It then elicits a prior from simulated experts, fits
  the model with PyMC, and ends with a posterior predictive check. ([#62](https://github.com/florence-bockting/elicito/pull/62))
- + The docs use the new API: a callable or PyMC model, flat `Elicit` settings,
    `fit(experts)`, `select=` and `pool=`.
  + The site renders the executed notebooks in the repository and runs nothing.
    Run a page with `make docs-notebooks PAGES=<page>.py`, then commit the `.py`
    page and its `.ipynb`. The CI checks that each notebook matches its page.
  + The tutorial "A prior for a panel of experts" uses `pool=el.partial(...)`.
    The new page *A panel of experts* explains the two stages.
  + New how-to guides: "Share a hyperparameter" and "Choose an initialization".
    The study of the initialization methods moved to *Further background*.
  + The human-growth example moved to the case studies.
  + The reference links in the notebooks, such as `[Elicit][elicito.elicit.Elicit]`,
    now work. A MkDocs hook, `scripts/notebook_autorefs.py`, passes them to autorefs. ([#68](https://github.com/florence-bockting/elicito/pull/68))


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
