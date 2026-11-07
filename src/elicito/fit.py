"""Result of a fit of an `Elicit` object"""

import functools
import inspect
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp  # type: ignore
import xarray as xr

from elicito import _panel, losses, models, plots
from elicito.parameters._base import seed_pair
from elicito.types import Parameter, Target

if TYPE_CHECKING:
    from elicito.elicit import Elicit


def _training_statistics(eliobj: "Elicit", replication: int) -> dict[str, tf.Tensor]:
    """Elicited statistics of the learned prior, with the draws of the fit"""
    seed = int(eliobj.results.history_stats.seed_replication.values[replication])
    trainer = {**eliobj.trainer, "seed": seed}
    prior_model = eliobj._learned_prior_model(replication, trainer, seed)
    training, *_ = models.one_forward_simulation(
        prior_model, eliobj.model, eliobj.targets, seed
    )
    return training


def _expert_statistics(eliobj: "Elicit") -> xr.Dataset:
    """Expert statistics of the fit; every replication holds the same values"""
    group = "expert" if "expert" in eliobj.results.children else "oracle"
    return eliobj.results[group].to_dataset().isel(replication=0)


def _best_replication(eliobj: "Elicit") -> int:
    """
    Find the replication with the lowest loss, for a parametric prior

    The loss is computed at the stored hyperparameters of each replication,
    with the draws of its fit. CMA-ES stores its best candidate, not the
    candidate of the last epoch, so the loss history does not give this
    loss. A replication whose loss is not finite is never the best.
    """
    stats = _expert_statistics(eliobj)
    expert = {
        tar["name"]: tf.constant(stats[tar["name"]].values) for tar in eliobj.targets
    }
    num_replications = len(eliobj.results.history_stats.seed_replication.values)
    total = np.full(num_replications, np.inf)
    for i in range(num_replications):
        loss, *_ = losses.total_loss(
            elicit_training=_training_statistics(eliobj, i),
            elicit_expert=expert,
            targets=eliobj.targets,
        )
        value = float(tf.squeeze(loss))
        if np.isfinite(value):
            total[i] = value
    return int(np.argmin(total))


_PLOTS = (
    "initialization",
    "loss",
    "hyperparameter",
    "prior_joint",
    "prior_marginals",
    "elicits",
    "marginals",
    "priorpredictive",
    "prior_averaging",
)


class _Plots:
    """Plot functions of `elicito.plots`, bound to one fit"""

    def __init__(self, eliobj: "Elicit", sample: Callable[[], Any]) -> None:
        self._eliobj = eliobj
        self._sample = sample

    def __getattr__(self, name: str) -> Callable[..., Any]:
        if name not in _PLOTS:
            msg = f"No plot named {name!r}. Available: {', '.join(_PLOTS)}."
            raise AttributeError(msg)
        func = getattr(plots, name)

        @functools.wraps(func)
        def bound(*args: Any, **kwargs: Any) -> Any:
            if "samples" in inspect.signature(func).parameters:
                kwargs.setdefault("samples", self._sample())
            return func(self._eliobj, *args, **kwargs)

        return bound

    def __dir__(self) -> list[str]:
        return list(_PLOTS)


def _prior_table(prior: xr.Dataset, reps: list[int]) -> xr.Dataset:
    """Summarize the prior draws of each parameter"""
    draws = prior.to_dataarray("parameter").isel(replication=reps)
    draws = draws.stack(sample=("batch", "draw"))
    return xr.Dataset(
        dict(
            mean=draws.mean("sample"),
            sd=draws.std("sample"),
            q05=draws.quantile(0.05, "sample").drop_vars("quantile"),
            q95=draws.quantile(0.95, "sample").drop_vars("quantile"),
        )
    )


def _hyper_table(rows: list[dict[str, float]], reps: list[int]) -> xr.Dataset:
    """Tabulate the hyperparameters, with their spread over replications"""
    values = xr.DataArray(
        [list(row.values()) for row in rows],
        dims=("replication", "hyperparameter"),
        coords=dict(replication=range(len(rows)), hyperparameter=list(rows[0])),
    )
    table = xr.Dataset(dict(estimate=values.isel(replication=reps)))
    if len(rows) > 1:
        table["rep_sd"] = values.std("replication", ddof=1)
        table["min"] = values.min("replication")
        table["max"] = values.max("replication")
    return table


def _expert_table(
    expert: xr.Dataset, elicited: xr.Dataset, reps: list[int]
) -> xr.Dataset | None:
    """Compare each expert statistic with the elicited statistic"""
    labels, given, learned = [], [], []
    # an oracle group also holds its prior draws; keep only statistics
    names = [name for name in expert.data_vars if name in elicited.data_vars]
    if not names:
        return None
    for name in names:
        values = np.ravel(expert[name].values)
        fitted = elicited[name].isel(replication=reps).mean("batch").values
        labels += [f"{name}[{k}]" for k in range(values.size)]
        given.append(values)
        learned.append(fitted.reshape(len(reps), -1))
    coords = dict(statistic=labels)
    expert_da = xr.DataArray(np.concatenate(given), dims="statistic", coords=coords)
    elicited_da = xr.DataArray(
        np.concatenate(learned, axis=1),
        dims=("replication", "statistic"),
        coords=dict(replication=reps, **coords),
    )
    return xr.Dataset(
        dict(expert=expert_da, elicited=elicited_da, error=elicited_da - expert_da)
    )


# name of each query, as the user calls its constructor
_TECHNIQUES = dict(
    quantiles="quantiles",
    pearson_correlation="correlation",
    identity="identity",
    custom="custom",
)


def _target_loss_table(
    loss: xr.Dataset, reps: list[int], targets: list[Target]
) -> xr.Dataset:
    """Tabulate the final loss term and the elicitation technique of each target"""
    terms = [name for name in loss.data_vars if name not in ("total_loss", "penalty")]
    final = loss[terms].isel(epoch=-1, replication=reps).drop_vars("epoch")
    table = xr.Dataset(dict(loss=final.to_dataarray("target")))
    technique = {tar["name"]: _TECHNIQUES[tar["query"]["name"]] for tar in targets}
    names = [technique[str(name)] for name in table["target"].values]
    return table.assign_coords(technique=("target", names))


_GROUPS = dict(
    hyperparameters=("Hyperparameters", "hyperparameter"),
    priors=("Priors", "parameter"),
    expert_fit=("Fit to expert", "statistic"),
    target_loss=("Loss by target", "target"),
    loss=("Final loss", "replication"),
)


def _table(ds: xr.Dataset, row_dim: str) -> str:
    """Format a dataset as a text table, one row per coordinate and replication"""
    split = "replication" in ds.dims and row_dim != "replication"
    reps = list(ds["replication"].values) if split else [None]
    columns = [str(name) for name in ds.data_vars]
    rep_column = len(reps) > 1
    labels = [str(label) for label in ds[row_dim].values]
    # a string coordinate on the rows, such as the technique, is a text column
    text = {
        str(name): [str(value) for value in coord.values]
        for name, coord in ds.coords.items()
        if name != row_dim and coord.dims == (row_dim,) and coord.dtype.kind in "OU"
    }
    pad = max(len(label) for label in labels)
    rows: list[tuple[str, list[str], list[float]]] = []
    for i, label in enumerate(labels):
        for r in reps:
            sub = ds.isel({row_dim: i})
            if r is not None:
                sub = sub.sel(replication=r)
            name = f"{label:<{pad}}  {r:>3}" if rep_column else label
            texts = [values[i] for values in text.values()]
            rows.append((name, texts, [float(sub[c]) for c in columns]))
    left = max(len(name) for name, _, _ in rows)
    text_widths = [max(len(c), *map(len, v)) + 2 for c, v in text.items()]
    widths = [max(9, len(c)) + 2 for c in columns]
    head = f"{'rep':>{left}}" if rep_column else " " * left
    head += "".join(f"{c:>{w}}" for c, w in zip(text, text_widths))
    lines = [head + "".join(f"{c:>{w}}" for c, w in zip(columns, widths))]
    for name, texts, values in rows:
        words = "".join(f"{t:>{w}}" for t, w in zip(texts, text_widths))
        cells = "".join(f"{v:>{w}.3g}" for v, w in zip(values, widths))
        lines.append(f"{name:<{left}}{words}{cells}")
    return "\n".join(lines)


def _format(tree: xr.DataTree) -> str:
    """Format a summary tree in the style of brms"""
    attrs = tree.attrs
    lines = [
        f"Method: {attrs['method']} ({attrs['epochs']} epochs)",
        f"Shown: replication {attrs['reps']} ({attrs['replication']})",
    ]
    for group, (title, row_dim) in _GROUPS.items():
        if group in tree.children:
            table = _table(tree[group].to_dataset(), row_dim)
            lines += ["", f"{title}:", table]
        if group == "expert_fit":
            lines += [
                f"{name}: draws, not statistics; see its loss in 'Loss by target'."
                for name in attrs.get("draw_targets", [])
            ]
    if "hyperparameters" in tree.children and "rep_sd" in tree["hyperparameters"]:
        lines += ["", "rep_sd, min and max are over all replications."]
    return "\n".join(lines)


class Fit:
    """Learned priors of one fit, with the methods to inspect them"""

    def __init__(self, eliobj: "Elicit") -> None:
        self._eliobj = eliobj
        self._samples: Any = None
        self._summary: xr.DataTree | None = None
        self.plot = _Plots(eliobj, self.sample)

    def __repr__(self) -> str:
        """Show the summary of the best replication; computed once"""
        if self._summary is None:
            self._summary = self.summary()
        return _format(self._summary)

    @property
    def eliobj(self) -> "Elicit":
        """Return the Elicit object that produced this fit"""
        return self._eliobj

    @property
    def results(self) -> Any:
        """Return the stored results as a DataTree"""
        return self._eliobj.results

    def best(self) -> int:
        """Return the index of the replication with the lowest loss"""
        if self._eliobj.trainer["method"] == "parametric_prior":
            return _best_replication(self._eliobj)
        loss = self.results["history_stats/loss"]["total_loss"]
        final = loss.isel(epoch=-1).values
        if not np.isfinite(final).any():
            msg = "No replication has a finite final loss."
            raise ValueError(msg)
        return int(np.nanargmin(np.where(np.isfinite(final), final, np.nan)))

    def hyperparameters(
        self, replication: int | Literal["best"] = "best"
    ) -> dict[str, float]:
        """Return the learned hyperparameters of one replication, by name"""
        index = self.best() if replication == "best" else replication
        return self._eliobj.hyperparameters(replication=index)

    def sample(
        self,
        num_samples: int | None = None,
        B: int | None = None,
        seed: int | None = None,
    ) -> Any:
        """Simulate from the learned prior; the default call is cached"""
        if num_samples is None and B is None and seed is None:
            if self._samples is None:
                self._samples = self._eliobj.sample()
            return self._samples
        return self._eliobj.sample(num_samples=num_samples, B=B, seed=seed)

    def summary(
        self, replication: int | Literal["best", "all"] = "best"
    ) -> xr.DataTree:
        """Return the learned priors and their fit to the expert"""
        n_reps = self.results["history_stats"].sizes["replication"]
        if replication == "all":
            reps = list(range(n_reps))
        else:
            reps = [self.best() if replication == "best" else replication]
        samples = self.sample()
        expert = _expert_statistics(self._eliobj).isel(batch=0)
        # an identity statistic holds all draws; see notes/TODO.md
        draw_targets = [
            tar["name"]
            for tar in self._eliobj.targets
            if tar["query"]["name"] == "identity"
        ]
        expert = expert.drop_vars(draw_targets)
        loss = self.results["history_stats/loss"].to_dataset()
        groups = dict(
            priors=_prior_table(samples["prior"].to_dataset(), reps),
            target_loss=_target_loss_table(loss, reps, self._eliobj.targets),
            loss=loss["total_loss"].isel(epoch=-1).to_dataset(name="final_loss"),
        )
        expert_fit = _expert_table(
            expert, samples["elicited_summary"].to_dataset(), reps
        )
        if expert_fit is not None:
            groups["expert_fit"] = expert_fit
        if self._eliobj.trainer["method"] == "parametric_prior":
            rows = [self.hyperparameters(r) for r in range(n_reps)]
            groups["hyperparameters"] = _hyper_table(rows, reps)
        tree = xr.DataTree.from_dict(groups)
        tree.attrs = dict(
            method=self._eliobj.trainer["method"],
            epochs=self._eliobj.trainer["epochs"],
            replication=str(replication),
            reps=reps,
            draw_targets=draw_targets,
        )
        return tree


class Panel:
    """Learned priors of a panel of experts"""

    def __init__(  # noqa: PLR0913
        self,
        experts: list[str],
        pool: Any,
        parameters: list[Parameter],
        fits: dict[str, Fit] | None = None,
        joint: Fit | None = None,
        tau: dict[str, float] | None = None,
    ) -> None:
        self.experts = experts
        self.pool = pool
        # one fit for each expert without pooling, else one joint fit
        self.fits = fits
        self.joint = joint
        self.tau = tau
        self._parameters = parameters
        self._hypers = {
            hyp["name"]: hyp
            for par in parameters
            for hyp in par["hyperparams"].values()
        }

    def __getitem__(self, expert: str) -> Fit:
        """Return the fit of one expert; only without pooling"""
        if self.fits is None:
            msg = (
                "A pooled panel has one joint fit, not one fit for each "
                "expert. Use panel.joint or panel.hyperparameters(expert)."
            )
            raise KeyError(msg)
        return self.fits[expert]

    def __repr__(self) -> str:
        """Show the pool, the spread and the hyperparameters of each expert"""
        pool = self.pool if isinstance(self.pool, str | None) else "partial"
        lines = [f"Panel of {len(self.experts)} experts, pool={pool!r}"]
        if self.tau is not None:
            spread = ", ".join(f"{name}={v:.3g}" for name, v in self.tau.items())
            lines.append(f"tau: {spread}")
        rows = ([] if self.joint is None else ["panel"]) + self.experts
        values = [self.hyperparameters(None if r == "panel" else r) for r in rows]
        table = xr.Dataset(
            {name: ("expert", [v[name] for v in values]) for name in values[0]},
            coords=dict(expert=rows),
        )
        lines += ["", "Hyperparameters:", _table(table, "expert")]
        return "\n".join(lines)

    def _index(self, expert: str) -> int:
        if expert not in self.experts:
            msg = f"Unknown expert {expert!r}. Known: {self.experts}."
            raise KeyError(msg)
        return self.experts.index(expert)

    def hyperparameters(
        self,
        expert: str | None = None,
        replication: int | Literal["best"] = "best",
    ) -> dict[str, float]:
        """Return the hyperparameters of the panel, or of one expert, by name"""
        if self.joint is None:
            if expert is None:
                msg = (
                    "A panel without pooling has no panel values. Give an "
                    "expert, for example panel.hyperparameters('expert 1')."
                )
                raise ValueError(msg)
            return self[expert].hyperparameters(replication)
        learned = self.joint.hyperparameters(replication)
        values = {name: learned[name] for name in self._hypers}
        if expert is None or self.tau is None:
            if expert is not None:
                self._index(expert)
            return values
        e = self._index(expert)
        for name in self.tau:
            hyp = self._hypers[name]
            offset = learned[_panel.suffix(name, e)]
            values[name] = float(
                hyp["constraint"](hyp["unconstrain"](values[name]) + offset)
            )
        return values

    def offsets(self, replication: int | Literal["best"] = "best") -> xr.Dataset:
        """Return the offset of each expert, on the unconstrained scale"""
        if self.joint is None or self.tau is None:
            msg = "Only pool=el.partial(...) has offsets."
            raise ValueError(msg)
        learned = self.joint.hyperparameters(replication)
        return xr.Dataset(
            {
                name: (
                    "expert",
                    [learned[_panel.suffix(name, e)] for e in range(len(self.experts))],
                )
                for name in self.tau
            },
            coords=dict(expert=self.experts),
        )

    def population_prior(self, num_samples: int = 10_000, seed: int = 0) -> xr.Dataset:
        """
        Sample the prior of the whole panel

        With partial pooling, each draw first draws an offset with the
        learned spread ``tau``. The prior is then wider than the prior of one
        expert, because it holds the disagreement in the panel.
        """
        if self.joint is None:
            msg = (
                "A panel without pooling has no population prior. Set "
                "pool='complete' or pool=el.partial(...)."
            )
            raise ValueError(msg)
        values = self.hyperparameters()
        tau = self.tau or {}
        seeds = tfp.random.split_seed(
            seed_pair(seed), n=len(tau) + len(self._parameters), salt="population"
        )
        # one offset for each name, shared by every parameter that uses it
        offsets = {
            name: tf.random.stateless_normal((num_samples,), seed=seeds[i]) * spread
            for i, (name, spread) in enumerate(tau.items())
        }
        draws = {}
        for i, par in enumerate(self._parameters):
            kwargs = {}
            for key, hyp in par["hyperparams"].items():
                value = tf.constant(values[hyp["name"]], tf.float32)
                if hyp["name"] in offsets:
                    value = hyp["constraint"](
                        hyp["unconstrain"](value) + offsets[hyp["name"]]
                    )
                kwargs[key] = value
            shifted = any(h["name"] in offsets for h in par["hyperparams"].values())
            shape = () if shifted else (num_samples,)
            sample = np.asarray(
                par["family"](**kwargs).sample(shape, seed=seeds[len(tau) + i])
            )
            extra = [f"{par['name']}_dim_{k}" for k in range(sample.ndim - 1)]
            dims = ("draw", *extra)
            draws[par["name"]] = (dims, sample)
        return xr.Dataset(draws)
