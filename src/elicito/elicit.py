"""
The Elicit object, which fits, samples and saves an elicitation method
"""

import os
import warnings
from collections import defaultdict
from collections.abc import Callable
from typing import Any

import joblib
import tensorflow as tf
import tensorflow_probability as tfp  # type: ignore
from joblib.externals.loky.backend.context import (  # type: ignore [import-untyped]
    get_context,
)

from elicito import (
    _checks,
    _outputs,
    _settings,
    _storage,
    initializers,
    models,
    optimizers,
    parameters,
    specs,
    utils,
)
from elicito._progress import SeedTable, run_in_worker
from elicito.fit import Fit
from elicito.parameters._base import _constraints
from elicito.parameters.priors import Priors
from elicito.types import (
    ExpertDict,
    Initializer,
    NFDict,
    Parameter,
    SamplingMethod,
    Target,
    Trainer,
)

tfd = tfp.distributions


def _default_initializer(
    optimizer: dict[str, Any], initializer: Initializer | None
) -> Initializer | None:
    """Use the default box of CMA-ES, and ignore a user initializer for it"""
    if optimizer["optimizer"] != optimizers.cmaes.CMAES:
        return initializer
    if initializer is not None:
        warnings.warn(
            f"optimizer='{optimizers.cmaes.CMAES}' ignores the initializer. "
            "Set the step size with cmaes=dict(sigma0=...).",
            UserWarning,
            stacklevel=3,
        )
    return initializers.initializer(method=SamplingMethod.cmaes)


def _dry_run(  # noqa: PLR0913
    model: dict[str, Any],
    parameters: list[Parameter],
    targets: list[Target],
    trainer: Trainer,
    initializer: Initializer,
    network: NFDict | None,
) -> tuple[dict[Any, Any], tf.Tensor, dict[Any, Any], dict[Any, Any], Any]:
    """
    Run generative model in forward mode for a single epoch

    Parameters
    ----------
    model
        User-input from [`model`][elicito.specs.model].

    parameters
        User-input from [`parameter`][elicito.specs.parameter].

    targets
        User-input from a target constructor such as
        [`quantiles`][elicito.specs.quantiles].

    trainer
        User-input from [`trainer`][elicito.specs.trainer].

    initializer
        User-input from [`initializer`][elicito.initializers.spec.initializer].

    network
        User-input from one of the methods implemented in the
        [`networks`][elicito.parameters.networks] module.

    Returns
    -------
    :
        (elicited_statistics, prior_samples, model_simulations,
        target_quantities, prior_model)
    """
    init_matrix_slice = (
        None
        if initializer is None
        else initializers.methods.resolve_init_method(initializer).dry_run_slice(
            initializer, parameters, trainer
        )
    )

    prior_model = Priors(
        ground_truth=False,
        init_matrix_slice=init_matrix_slice,
        trainer=trainer,
        parameters=parameters,
        network=network,
        expert=None,  # type: ignore
        seed=trainer["seed"],
    )

    (elicited_statistics, prior_samples, model_simulations, target_quantities) = (
        models.one_forward_simulation(
            prior_model=prior_model, model=model, targets=targets, seed=trainer["seed"]
        )
    )

    return (
        elicited_statistics,
        prior_samples,
        model_simulations,
        target_quantities,
        prior_model,
    )


class Elicit:
    """
    Configure the elicitation method
    """

    def __init__(  # noqa: PLR0913
        self,
        model: dict[str, Any],
        targets: list[Target],
        parameters: list[Parameter],
        expert: ExpertDict,
        *,
        network: NFDict | None = None,
        initializer: Initializer | None = None,
        optimizer: Any = "auto",
        cmaes: dict[str, Any] | None = None,
        seeds: tuple[int, ...] = (0,),
        epochs: int = 600,
        B: int = 128,
        num_samples: int = 200,
        kappa: float = 0.0,
        verbose: bool = True,
        dry_run: bool = True,
    ):
        """
        Specify the elicitation method

        Parameters
        ----------
        model
            specification of generative model using [`model`][elicito.specs.model].

        targets
            list of target quantities, specified with a target constructor
            such as [`quantiles`][elicito.specs.quantiles].

        parameters
            list of model parameters specified with [`parameter`][elicito.specs.parameter].

        expert
            provide input data from expert or simulate data from oracle with
            either the ``data`` or ``simulator`` method of the
            [`Expert`][elicito.specs.Expert] module.

        network
            normalizing flow from [`networks`][elicito.parameters.networks].
            A network means a deep prior: one joint prior of all parameters.
            Without a network, elicito learns one parametric prior for each
            parameter.

        initializer
            specification of initialization settings using
            [`initializer`][elicito.initializers.spec.initializer].
            Used for parametric priors only. CMA-ES ignores it with a warning.

        optimizer
            ``"auto"`` uses CMA-ES for parametric priors and Adam for a deep
            prior. ``"cmaes"`` replaces the gradient descent by a CMA-ES search
            ([`cma_training`][elicito.optimizers.cmaes.cma_training]). It needs
            no gradient, and it can leave a local basin. It is available for
            parametric priors only. ``"adam"`` uses Adam with a learning rate
            of 0.1. For other settings, give a
            [`tf.keras` optimizer](https://www.tensorflow.org/api_docs/python/tf/keras/optimizers),
            for example ``tf.keras.optimizers.Adam(learning_rate=0.01)``. Each
            run starts with a new optimizer with these settings.

        cmaes
            settings of the CMA-ES search. ``sigma0`` is the first step size
            on the unconstrained scale, one number for every coordinate. Give
            a dictionary ``{hyperparameter name: step size}`` if the
            hyperparameters do not live on the same scale. A name that is not
            given gets the default step size. ``popsize`` is the number of
            candidates in one generation. Ignored with a warning if the
            optimizer is not CMA-ES.

        seeds
            seeds of the training. One replication runs for each seed.

        epochs
            number of epochs. With CMA-ES, ``epochs`` is the budget in forward
            simulations, not the number of generations.

        B
            batch size.

        num_samples
            number of samples from the prior(s).

        kappa
            weight of the spread penalty. The penalty keeps a prior away from
            a point mass. Use it when a hyperparameter is not identified by
            the elicited statistics. A value of 0 disables the penalty.

        verbose
            show the training progress.

        dry_run
            run the generative model once in forward mode before the
            training. The print method then shows the shapes of all tensors.

        Returns
        -------
        eliobj :
            specification of all settings to run the elicitation workflow and
            fit the eliobj.

        Raises
        ------
        AssertionError
            ``ground_truth`` for simulating expert data has the wrong
            dimension.

        ValueError
            ``expert`` answers do not match the targets. The keys must be the
            target names.

        ValueError
            ``optimizer`` is not ``"auto"``, ``"cmaes"``, ``"adam"`` or a
            tf.keras optimizer.

            ``optimizer="cmaes"`` is given with a ``network``.

            with a ``network``, ``initializer`` is not None.

            with a ``network``, ``num_params`` of the network does not match
            the number of parameters.

            without a ``network``, ``initializer`` is None and the optimizer
            is not CMA-ES.

            multiple hyperparameters have the same name but are not shared by
            setting ``shared = True``.

            if ``hyperparams`` is specified in section ``initializer`` and a
            hyperparameter name (key in hyperparams dict) does not match any
            hyperparameter name specified in [`hyper`][elicito.specs.hyper].

        NotImplementedError
            [network] Currently only the standard normal distribution is
            implemented as base distribution. See
            [GitHub issue #35](https://github.com/florence-bockting/prior_elicitation/issues/35).

        """  # noqa: E501
        seeds = _settings.seeds(seeds)
        self.settings: dict[str, Any] = dict(
            network=network,
            initializer=initializer,
            optimizer=optimizer,
            cmaes=cmaes,
            seeds=seeds,
            epochs=epochs,
            B=B,
            num_samples=num_samples,
            kappa=kappa,
            verbose=verbose,
            dry_run=dry_run,
        )
        trainer = _settings.trainer(
            network, seeds, epochs, B, num_samples, kappa, verbose
        )
        optimizer = _settings.optimizer(optimizer, cmaes, network)
        meta_settings = specs.meta_settings(dry_run=dry_run)
        initializer = _default_initializer(optimizer, initializer)

        _checks.check_elicit(
            model,
            parameters,
            targets,
            expert,
            trainer,
            optimizer,
            network,
            initializer,
            meta_settings,
        )

        self.model = model
        self.parameters = parameters
        self.targets = targets
        self.expert = expert
        self.trainer = trainer
        self.optimizer = optimizer
        self.network = network
        self.initializer = initializer
        self.meta_settings = meta_settings

        self.temp_history: list[dict[str, Any]] = []
        self.temp_results: list[dict[str, Any]] = []

        # helper for subsequent checks
        self.dry_run = self.meta_settings["dry_run"]
        # overwrite global seed
        utils.SEED = self.trainer["seed"]

        # set seed
        tf.random.set_seed(utils.SEED)

        if self.dry_run:
            (
                self.dry_elicits,
                self.dry_priors,
                self.dry_modelsims,
                self.dry_targets,
                self.dry_prior_model,
            ) = _dry_run(
                self.model,
                self.parameters,
                self.targets,
                self.trainer,
                self.initializer,  # type: ignore
                self.network,
            )

    def __str__(self) -> str:  # noqa: PLR0912
        """Return a readable summary of the object."""
        queries = {tar["name"]: tar["query"]["name"] for tar in self.targets}
        names_str = "\n".join(
            f"  - {name} -> {query}" for name, query in queries.items()
        )

        if hasattr(self, "results"):
            targets_str = names_str
        elif len(self.temp_results) != 0:
            res = self.temp_results[0]
            targets_str = "\n".join(
                f"  - {name} {tuple(res['target_quantities'][name].shape)} -> "
                f"{query} {tuple(res['elicited_statistics'][name].shape)}"
                for name, query in queries.items()
            )
        # unfitted eliobj with shape information due to dry run
        elif self.dry_run:
            targets_str = "\n".join(
                f"  - {name} {tuple(self.dry_targets[name].shape)} -> "
                f"{query} {tuple(self.dry_elicits[name].shape)}"
                for name, query in queries.items()
            )
        # unfitted eliobj without shape information
        else:
            targets_str = names_str

        if self.optimizer["optimizer"] == optimizers.cmaes.CMAES:
            sigma0 = self.optimizer.get("sigma0", optimizers.cmaes.DEFAULT_SIGMA0)
            if isinstance(sigma0, dict):
                # a step size per hyperparameter is too long for one line
                sigma0 = f"{len(sigma0)} values"
            opt_str = f"{optimizers.cmaes.CMAES}(sigma0={sigma0})"
        else:
            opt_name = self.optimizer["optimizer"].__name__
            opt_lr = self.optimizer["learning_rate"]
            opt_str = f"{opt_name}(lr={opt_lr})"

        get_num_hyperpar: int | str
        if hasattr(self, "results") and self.trainer["method"] == "deep_prior":
            get_num_hyperpar = utils.compute_num_weights(
                self.results[0]["num_NN_weights"]  # type: ignore
            )

        if (self.trainer["method"] == "deep_prior") and (self.dry_run):
            trainable_vars = self.dry_prior_model.init_priors.trainable_variables
            num_NN_weights = [
                trainable_vars[i].shape for i in range(len(trainable_vars))
            ]
            get_num_hyperpar = utils.compute_num_weights(num_NN_weights)

        elif self.trainer["method"] == "parametric_prior":
            get_num_hyperpar = sum(
                [
                    len(self.parameters[i]["hyperparams"])
                    for i in range(len(self.parameters))
                ]
            )
        else:
            get_num_hyperpar = "?"
            print("Number of hyperparameter in model can't be computed.")

        summary = (
            f"Model hyperparameters: {get_num_hyperpar}\n"
            f"Model parameters: {len(self.parameters)}\n"
            "Targets -> Elicited summaries (loss components)"
            f"{': ' + str(len(self.dry_elicits)) if self.dry_run else ''}\n"
            f"{targets_str}\n"
            f"Prior samples: {self.trainer['num_samples']}"
            f"{' ' + str(tuple(self.dry_priors.shape)) if self.dry_run else ''}\n"
            f"Batch size: {self.trainer['B']}\n"
            f"Epochs: {self.trainer['epochs']}\n"
            f"Method: {self.trainer['method']}\n"
            f"Seed: {self.trainer['seed']}\n"
            f"Optimizer: {opt_str}\n"
        )
        if self.trainer["method"] == "parametric_prior":
            if self.initializer is not None:
                summary += (
                    f"Initializer: (method: {self.initializer['method']}, "
                    f"iterations: {self.initializer['iterations']})\n"
                )
            else:
                summary += "Initializer: None\n"
        elif self.network is not None:
            summary += f"Network: {self.network['inference_network'].__name__}\n"
        else:
            summary += "Network: None\n"

        return summary

    def __repr__(self) -> str:
        """Return a readable representation of the object."""
        return self.__str__()

    def fit(
        self,
        overwrite: bool = False,
    ) -> Fit:
        """
        Fit the eliobj and learn prior distributions.

        Parameters
        ----------
        overwrite
            If the eliobj was already fitted and the user wants to refit it,
            the user is asked whether they want to overwrite the previous
            fitting results. Setting ``overwrite=True`` allows the user to
            force overfitting without being prompted.

        Returns
        -------
        :
            learned priors, with methods to inspect them

        Raises
        ------
        ValueError
            The eliobj is already fitted and ``overwrite`` is ``False``.

        Examples
        --------
        >>> eliobj.fit()  # doctest: +SKIP

        >>> eliobj.fit(overwrite=True)  # doctest: +SKIP

        """
        # set seed
        tf.random.set_seed(self.trainer["seed"])

        # check whether elicit object is already fitted
        if hasattr(self, "results"):
            if not overwrite:
                msg = (
                    "eliobj is already fitted. Use overwrite=True to fit it "
                    "again and replace the results."
                )
                raise ValueError(msg)
            delattr(self, "results")

        self.temp_results = []
        self.temp_history = []

        seeds = self.settings["seeds"]
        # one seed runs in this process
        if len(seeds) == 1:
            results, history = self.workflow(seeds[0])
            # include seed information into results
            results["seed"] = seeds[0]
            # save results in list attribute
            self.temp_history.append(history)
            self.temp_results.append(results)
        # run multiple replications
        else:
            # run training simultaneously for multiple seeds. Live tables from
            # several processes overwrite each other, so the workers send
            # their tables to the parent. It shows one row for each seed.
            # The loky context starts the queue server without fork, which
            # can deadlock in a multi-threaded process.
            with get_context("loky").Manager() as manager:
                queue = manager.Queue()
                table = SeedTable(seeds, queue, disable=self.trainer["progress"] == 0)
                try:
                    res = joblib.Parallel(n_jobs=min(len(seeds), os.cpu_count() or 1))(
                        joblib.delayed(run_in_worker)(self.workflow, seed, row, queue)
                        for row, seed in enumerate(seeds)
                    )
                finally:
                    table.close()

            for i, seed in enumerate(seeds):
                self.temp_results.append(res[i][0])
                self.temp_history.append(res[i][1])
                self.temp_results[i]["seed"] = seed

        self.results = _outputs.create_datatree(
            self.temp_history,
            self.temp_results,
            self.trainer,
            self.parameters,
            self.expert,
        )

        delattr(self, "temp_history")
        delattr(self, "temp_results")
        return Fit(self)

    def sample(
        self,
        num_samples: int | None = None,
        B: int | None = None,
        seed: int | None = None,
    ) -> Any:
        """
        Simulate from the learned prior

        The learned trainable variables, the generative model and the
        parameter definitions reproduce the prior samples, the model
        simulations, the target quantities and the elicited summaries. The
        method runs one forward pass per replication.

        Parameters
        ----------
        num_samples
            number of prior samples per batch. Default is the value used
            for training.

        B
            batch size. Default is the value used for training.

        seed
            seed of the forward pass. Default is the seed of the
            corresponding replication. With the default, and with the
            training values for **num_samples** and **B**, the samples
            equal those of the last training epoch.

        Returns
        -------
        :
            xr.DataTree with the groups prior, model, target_quantity and
            elicited_summary.

        Raises
        ------
        AttributeError
            eliobj has not been fitted yet.

        ValueError
            The stored weights do not match the trainable variables.

        Examples
        --------
        >>> samples = eliobj.sample(num_samples=1_000)  # doctest: +SKIP
        >>> el.plots.prior_marginals(samples)  # doctest: +SKIP
        """
        if not hasattr(self, "results"):
            msg = "No results found. Run 'eliobj.fit()' before 'eliobj.sample()'."
            raise AttributeError(msg)

        seeds = self.results.history_stats.seed_replication.values

        simulated = []
        for i, replication_seed in enumerate(seeds):
            run_seed = int(replication_seed) if seed is None else int(seed)
            trainer = dict(self.trainer)
            trainer["seed"] = run_seed
            if num_samples is not None:
                trainer["num_samples"] = num_samples
            if B is not None:
                trainer["B"] = B

            prior_model = self._learned_prior_model(i, trainer, run_seed)

            tf.random.set_seed(run_seed)
            (elicits, prior_sim, model_sim, target_quants) = models.simulate_and_elicit(
                prior_model=prior_model,
                model=self.model,
                targets=self.targets,
                seed=run_seed,
            )
            simulated.append(
                dict(
                    prior_samples=prior_sim,
                    model_samples=model_sim,
                    target_quantities=target_quants,
                    elicited_statistics=elicits,
                )
            )

        return _outputs.create_sample_tree(simulated, self.parameters)

    def _learned_prior_model(
        self, replication: int, trainer: dict[str, Any], seed: int
    ) -> Any:
        """Rebuild the prior model of one replication with its learned weights"""
        weights = self.results["learned_weights"].to_dataset()
        method = parameters.methods.get_method(self.trainer["method"])

        # the build step reads an initial value for every hyperparameter.
        # The learned values overwrite them below, so any number does.
        prior_model = parameters.priors.Priors(
            ground_truth=False,
            init_matrix_slice=defaultdict(lambda: tf.constant(0.0)),
            trainer=trainer,  # type: ignore [arg-type]
            parameters=self.parameters,
            network=self.network,
            expert=self.expert,
            seed=seed,
        )
        variables = method.trainable_variables(prior_model)
        if len(variables) != len(weights.data_vars):
            msg = (
                f"The model has {len(variables)} trainable variables but"
                f" {len(weights.data_vars)} are stored in the results."
                " The results belong to a different model specification."
            )
            raise ValueError(msg)
        for j, variable in enumerate(variables):
            variable.assign(weights[f"weight_{j}"].sel(replication=replication).values)
        return prior_model

    def hyperparameters(self, replication: int = 0) -> dict[str, float]:
        """
        Return the learned hyperparameter values

        The values belong to the prior that the fit returns, on the scale of
        the prior family.

        Parameters
        ----------
        replication
            index of the replication

        Returns
        -------
        :
            learned value of each hyperparameter, by name

        Raises
        ------
        AttributeError
            eliobj has not been fitted yet.

        ValueError
            The method is not ``"parametric_prior"``.

        Examples
        --------
        >>> eliobj.hyperparameters()  # doctest: +SKIP
        """
        if not hasattr(self, "results"):
            msg = (
                "No results found. Run 'eliobj.fit()' before"
                " 'eliobj.hyperparameters()'."
            )
            raise AttributeError(msg)
        if self.trainer["method"] != "parametric_prior":
            msg = "Only method='parametric_prior' has hyperparameters."
            raise ValueError(msg)

        seed = int(self.results.history_stats.seed_replication.values[replication])
        prior_model = self._learned_prior_model(replication, dict(self.trainer), seed)
        constraints = _constraints(self.parameters)
        values = {}
        for var in prior_model.trainable_variables:
            name = var.name[:-2].split(".")[1]
            values[name] = float(constraints[name](var.numpy()))
        return values

    def save(
        self,
        name: str | None = None,
        file: str | None = None,
        overwrite: bool = False,
    ) -> None:
        """
        Save data on disk

        Parameters
        ----------
        name
            file name used to store the eliobj. Saving is done
            according to the following rule: ``./{method}/{name}_{seed}.pkl``
            with 'method' and 'seed' being arguments of
            [`trainer`][elicito.specs.trainer].

        file
            user-specific path for saving the eliobj. If file is specified
            **name** must be ``None``.

        overwrite
            If already a fitted object exists in the same path, the user is
            asked whether the eliobj should be refitted and the results
            overwritten.
            With the ``overwrite`` argument, you can disable this
            behavior. In this case the results are automatically overwritten
            without prompting the user.

        Raises
        ------
        AssertionError
            ``name`` and ``file`` can't be specified simultaneously.

        Examples
        --------
        >>> eliobj.save(name="toymodel")  # doctest: +SKIP

        >>> eliobj.save(file="res/toymodel", overwrite=True)  # doctest: +SKIP

        """
        return _storage.save(self, name=name, file=file, overwrite=overwrite)

    @classmethod
    def load(cls, file: str) -> "Elicit":
        """
        Load a saved ``eliobj`` from specified path

        Parameters
        ----------
        file
            path where ``eliobj`` object is saved.

        Returns
        -------
        eliobj :
            loaded ``eliobj`` object.

        Examples
        --------
        >>> eliobj = el.Elicit.load("res/toymodel.pkl")  # doctest: +SKIP

        """
        storage = _storage.read_storage(file)
        eliobj = cls(
            model=storage["model"],
            parameters=storage["parameters"],
            targets=storage["targets"],
            expert=storage["expert"],
            **storage["settings"],
        )

        # add results if already fitted
        if "results" in storage:
            eliobj.results = storage["results"]
        else:
            eliobj.temp_history = storage["temp_history"]
            eliobj.temp_results = storage["temp_results"]

        return eliobj

    def update(self, **kwargs: Any) -> None:
        """
        Update the inputs of the Elicit object

        Updating an eliobj resets its results.

        Parameters
        ----------
        **kwargs
            new values for the arguments of [`Elicit`][elicito.elicit.Elicit],
            for example ``epochs=200`` or ``optimizer="adam"``.

        Raises
        ------
        ValueError
            a keyword is not an argument of [`Elicit`][elicito.elicit.Elicit].

        Examples
        --------
        >>> eliobj.update(epochs=200)  # doctest: +SKIP

        """
        inputs = dict(
            model=self.model,
            targets=self.targets,
            parameters=self.parameters,
            expert=self.expert,
            **self.settings,
        )
        unknown = set(kwargs) - set(inputs)
        if unknown:
            msg = f"{sorted(unknown)} are not arguments of Elicit."
            raise ValueError(msg)
        inputs.update(kwargs)
        # CMA-ES ignores an initializer: drop it, unless it is given here
        new_optimizer = inputs["optimizer"]
        if "initializer" not in kwargs and isinstance(new_optimizer, str):
            if new_optimizer == optimizers.cmaes.CMAES or (
                new_optimizer == _settings.AUTO and inputs["network"] is None
            ):
                inputs["initializer"] = None

        # the constructor runs the checks before self changes
        updated = type(self)(**inputs)
        self.__dict__.clear()
        self.__dict__.update(updated.__dict__)
        if kwargs:
            print("INFO: Results have been reset.")

    def workflow(self, seed: int) -> tuple[Any, ...]:
        """
        Build the main workflow of the prior elicitation method.

        Get expert data, initialize method, run optimization.
        Results are returned for further post-processing.

        Parameters
        ----------
        seed
            seed information used for reproducing results.

        Returns
        -------
        :
            results and history object of the optimization process.

        """
        # overwrite global seed
        # TODO test correct seed usage for parallel processing
        utils.SEED = seed

        # get expert data; use trainer seed
        # (and not seed from list)
        expert_elicits, expert_prior = utils.get_expert_data(
            self.trainer,
            self.model,
            self.targets,
            self.expert,
            self.parameters,
            self.network,
            self.trainer["seed"],
        )

        # the prior draws its samples with the seed in the trainer, so each
        # replication needs a trainer with its own seed
        trainer: Trainer = {**self.trainer, "seed": seed}

        # initialization of hyperparameter
        (init_prior_model, loss_list, init_matrix) = initializers.methods.init_prior(
            expert_elicits,
            self.initializer,
            self.parameters,
            trainer,
            self.optimizer,
            self.model,
            self.targets,
            self.network,
            self.expert,
            seed,
            self.trainer["progress"],
        )
        # run dag with optimal set of initial values
        # save results in corresp. attributes

        # the optimizer is either a tf.keras class, or the name of a
        # derivative-free search
        extra: dict[str, Any] = {}
        fit_method: Callable[..., tuple[dict[Any, Any], dict[Any, Any]]]
        if self.optimizer["optimizer"] == optimizers.cmaes.CMAES:
            fit_method = optimizers.cmaes.cma_training
            if self.initializer is not None:
                extra["default_sigma0"] = optimizers.cmaes.box_step_size(
                    initializers.methods.resolve_init_method(self.initializer),
                    self.initializer,
                    self.parameters,
                )
        else:
            fit_method = optimizers.sgd.sgd_training

        history, results = fit_method(
            expert_elicits,
            init_prior_model,
            trainer,
            self.optimizer,
            self.model,
            self.targets,
            self.parameters,
            seed,
            self.trainer["progress"],
            **extra,
        )
        # add some additional results
        results["expert_elicited_statistics"] = expert_elicits
        try:
            self.expert["ground_truth"]
        except KeyError:
            pass
        else:
            results["expert_prior_samples"] = expert_prior

        if self.trainer["method"] == "parametric_prior":
            results["init_loss_list"] = loss_list
            results["init_matrix"] = init_matrix

        return tuple((results, history))
