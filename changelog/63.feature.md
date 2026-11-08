`eliobj.fit()` now returns a `Fit` object with the learned priors.
`fit.best()` gives the replication with the lowest loss, and
`fit.hyperparameters()` the learned values of that replication.
`fit.sample()` caches its default call. `fit.plot.<name>()` calls the plot
functions of `elicito.plots` with the cached samples. `fit.summary()` returns
the hyperparameters, the priors, the fit to the expert and the loss of each
target as an `xr.DataTree`, for the best replication, one replication or all
of them. `print(fit)` shows this summary as text tables.
