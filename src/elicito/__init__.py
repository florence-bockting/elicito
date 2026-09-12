"""
A Python package for learning prior distributions based on expert knowledge
"""

import importlib.metadata
from collections import defaultdict
from collections.abc import Callable
from types import SimpleNamespace
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
    cmaes,
    elicit,
    initialization,
    losses,
    methods,
    networks,
    optimization,
    plots,
    simulations,
    targets,
    types,
    utils,
    warmstart,
)
from elicito._progress import SeedTable, run_in_worker
from elicito.elicit import (
    expert,
    hyper,
    initializer,
    meta_settings,
    model,
    optimizer,
    parameter,
    queries,
    target,
    trainer,
)
from elicito.types import (
    ExpertDict,
    Initializer,
    MetaSettings,
    NFDict,
    Parallel,
    Parameter,
    SamplingMethod,
    Target,
    Trainer,
)

tfd = tfp.distributions

tf.get_logger().setLevel("ERROR")

__version__ = importlib.metadata.version("elicito")

__all__ = [
    "Elicit",
    "cmaes",
    "expert",
    "hyper",
    "initialization",
    "initializer",
    "losses",
    "meta_settings",
    "model",
    "networks",
    "optimization",
    "optimizer",
    "parameter",
    "plots",
    "queries",
    "simulations",
    "target",
    "targets",
    "trainer",
    "types",
    "utils",
    "warmstart",
]

# global variable (gets overwritten by user-defined
# seed in Elicit object)
SEED = 0


def _default_initializer(
    optimizer: dict[str, Any], initializer: Initializer | None
) -> Initializer | None:
    """Use the default box of CMA-ES if no initializer is given"""
    if initializer is None and optimizer["optimizer"] == cmaes.CMAES:
        return elicit.initializer(method=SamplingMethod.cmaes)
    return initializer


class Elicit:
    """
    Configure the elicitation method
    """

    def __init__(  # noqa: PLR0913
        self,
        model: dict[str, Any],
        parameters: list[Parameter],
        targets: list[Target],
        expert: ExpertDict,
        trainer: Trainer,
        optimizer: dict[str, Any],
        network: NFDict | None = None,
        initializer: Initializer | None = None,
        meta_settings: MetaSettings | None = None,
    ):
        """
        Specify the elicitation method

        Parameters
        ----------
        model
            specification of generative model using [`model`][elicito.elicit.model].

        parameters
            list of model parameters specified with [`parameter`][elicito.elicit.parameter].

        targets
            list of target quantities specified with [`target`][elicito.elicit.target].

        expert
            provide input data from expert or simulate data from oracle with
            either the ``data`` or ``simulator`` method of the
            [`Expert`][elicito.elicit.Expert] module.

        trainer
            specification of training settings and meta-information for
            workflow using [`trainer`][elicito.elicit.trainer].

        optimizer
            specification of SGD optimizer and its settings using
            [`optimizer`][elicito.elicit.optimizer].

        network
            specification of neural network using a method implemented in
            [`networks`][elicito.networks].
            Only required for ``deep_prior`` method.

        initializer
            specification of initialization settings using
            [`initializer`][elicito.elicit.initializer].
            Only required for ``parametric_prior`` method. With
            ``optimizer="cmaes"``, ``None`` means ``initializer(method="cmaes")``.

        meta_settings
            dictionary of meta settings for the elicitation workflow. See
            [`meta_settings`][elicito.types.MetaSettings] for available options.

        Returns
        -------
        eliobj :
            specification of all settings to run the elicitation workflow and
            fit the eliobj.

        Raises
        ------
        AssertionError
            ``expert`` data are not in the required format. Correct specification of
            keys can be checked using
            [`get_expert_datformat`][elicito.utils.get_expert_datformat]

            Dimensionality of ``ground_truth`` for simulating expert data, must be
            the same as the number of model parameters.

        ValueError
            if ``method = "deep_prior"``, ``network`` can't be None and ``initialization``
            should be None.

            if ``method="deep_prior"``, ``num_params`` as specified in the ``network_specs``
            argument (section: network) does not match the number of parameters
            specified in the parameters section.

            if ``method="parametric_prior"``, ``network`` should be None and
            ``initialization`` can't be None.

            if ``method ="parametric_prior" and multiple hyperparameter have
            the same name but are not shared by setting ``shared = True``."

            if ``hyperparams`` is specified in section ``initializer`` and a
            hyperparameter name (key in hyperparams dict) does not match any
            hyperparameter name specified in [`hyper`][elicito.elicit.hyper].

        NotImplementedError
            [network] Currently only the standard normal distribution is
            implemented as base distribution. See
            [GitHub issue #35](https://github.com/florence-bockting/prior_elicitation/issues/35).

        """  # noqa: E501
        if meta_settings is None:
            meta_settings = elicit.meta_settings()

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
        globals()["SEED"] = self.trainer["seed"]

        # set seed
        tf.random.set_seed(SEED)

        if self.dry_run:
            (
                self.dry_elicits,
                self.dry_priors,
                self.dry_modelsims,
                self.dry_targets,
                self.dry_prior_model,
            ) = utils.dry_run(
                self.model,
                self.parameters,
                self.targets,
                self.trainer,
                self.initializer,  # type: ignore
                self.network,
            )

    def __str__(self) -> str:  # noqa: PLR0912
        """Return a readable summary of the object."""
        names_str = "\n".join(
            f"  - {self.targets[tar]['name']} -> {eli}"
            for tar, eli in zip(
                range(len(self.targets)),
                utils.get_expert_datformat(self.targets),
            )
        )

        if hasattr(self, "results"):
            targets_str = names_str
        elif len(self.temp_results) != 0:
            targets_str = "\n".join(
                f"  - {k1} {tuple(self.temp_results[0]['target_quantities'][k1].shape)} -> "  # noqa: E501
                f"{k2} {tuple(self.temp_results[0]['elicited_statistics'][k2].shape)}"
                for k1, k2 in zip(
                    self.temp_results[0]["target_quantities"],
                    self.temp_results[0]["elicited_statistics"],
                )
            )
        # unfitted eliobj with shape information due to dry run
        elif self.dry_run:
            targets_str = "\n".join(
                f"  - {k1} {tuple(self.dry_targets[k1].shape)} -> "
                f"{k2} {tuple(self.dry_elicits[k2].shape)}"
                for k1, k2 in zip(self.dry_targets, self.dry_elicits)
            )
        # unfitted eliobj without shape information
        else:
            targets_str = names_str

        if self.optimizer["optimizer"] == cmaes.CMAES:
            sigma0 = self.optimizer.get("sigma0", cmaes.DEFAULT_SIGMA0)
            if isinstance(sigma0, dict):
                # a step size per hyperparameter is too long for one line
                sigma0 = f"{len(sigma0)} values"
            opt_str = f"{cmaes.CMAES}(sigma0={sigma0})"
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
        parallel: Parallel | None = None,
    ) -> None:
        """
        Fit the eliobj and learn prior distributions.

        Parameters
        ----------
        overwrite
            If the eliobj was already fitted and the user wants to refit it,
            the user is asked whether they want to overwrite the previous
            fitting results. Setting ``overwrite=True`` allows the user to
            force overfitting without being prompted.

        parallel
            specify parallelization settings if multiple trainings should run
            in parallel. See [`parallel`][elicito.utils.parallel].

        Raises
        ------
        ValueError
            The eliobj is already fitted and ``overwrite`` is ``False``.

        Examples
        --------
        >>> eliobj.fit()  # doctest: +SKIP

        >>> eliobj.fit(overwrite=True)  # doctest: +SKIP

        >>> eliobj.fit(parallel=el.utils.parallel(runs=4))  # doctest: +SKIP

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

        # run single time if no parallelization is required
        if parallel is None:
            results, history = self.workflow(self.trainer["seed"])
            # include seed information into results
            results["seed"] = self.trainer["seed"]
            # save results in list attribute
            self.temp_history.append(history)
            self.temp_results.append(results)
        # run multiple replications
        else:
            # create a list of seeds if not provided
            if parallel["seeds"] is None:
                # generate seeds
                seeds = [
                    int(s) for s in tfd.Uniform(0, 999999).sample(parallel["runs"])
                ]
            else:
                seeds = parallel["seeds"]

            # run training simultaneously for multiple seeds. Live tables from
            # several processes overwrite each other, so the workers send
            # their tables to the parent. It shows one row for each seed.
            # The loky context starts the queue server without fork, which
            # can deadlock in a multi-threaded process.
            with get_context("loky").Manager() as manager:
                queue = manager.Queue()
                table = SeedTable(seeds, queue, disable=self.trainer["progress"] == 0)
                try:
                    res = joblib.Parallel(n_jobs=parallel["cores"])(
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

        weights = self.results["learned_weights"].to_dataset()
        seeds = self.results.history_stats.seed_replication.values
        method = methods.get_method(self.trainer["method"])

        simulated = []
        for i, replication_seed in enumerate(seeds):
            run_seed = int(replication_seed) if seed is None else int(seed)
            trainer = dict(self.trainer)
            trainer["seed"] = run_seed
            if num_samples is not None:
                trainer["num_samples"] = num_samples
            if B is not None:
                trainer["B"] = B

            # the build step reads an initial value for every hyperparameter.
            # The learned values overwrite them below, so any number does.
            prior_model = simulations.Priors(
                ground_truth=False,
                init_matrix_slice=defaultdict(lambda: tf.constant(0.0)),
                trainer=trainer,  # type: ignore [arg-type]
                parameters=self.parameters,
                network=self.network,
                expert=self.expert,
                seed=run_seed,
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
                variable.assign(weights[f"weight_{j}"].sel(replication=i).values)

            tf.random.set_seed(run_seed)
            (elicits, prior_sim, model_sim, target_quants) = utils.simulate_and_elicit(
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
            [`trainer`][elicito.elicit.trainer].

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
        return utils.save(self, name=name, file=file, overwrite=overwrite)

    def update(self, **kwargs: dict[Any, Any]) -> None:
        """
        Update attributes of Elicit object

        Method for updating the attributes of the Elicit class. Updating
        an eliobj leads to an automatic reset of results.

        Parameters
        ----------
        **kwargs
            keyword argument used for updating an attribute of Elicit class.
            Key must correspond to one attribute of the class and value refers
            to the updated value.

        Raises
        ------
        ValueError
            key of provided keyword argument is not an eliobj attribute. Please
            check `dir(eliobj)`.

        Examples
        --------
        >>> eliobj.update(parameter=updated_parameter_dict)  # doctest: +SKIP

        """
        # check that arguments exist as eliobj attributes
        for key in kwargs:
            if str(key) not in [
                "model",
                "parameters",
                "targets",
                "expert",
                "trainer",
                "optimizer",
                "network",
                "initializer",
            ]:
                msg = (
                    f"{key=} is not an eliobj attribute. "
                    + "Use dir() to check for attributes.",
                )
                raise ValueError(msg)

        # create first test variables
        test = SimpleNamespace(
            model=self.model,
            parameters=self.parameters,
            targets=self.targets,
            expert=self.expert,
            trainer=self.trainer,
            optimizer=self.optimizer,
            network=self.network,
            initializer=self.initializer,
            meta_settings=self.meta_settings,
        )

        for key, value in kwargs.items():
            setattr(test, key, value)

        test.initializer = _default_initializer(test.optimizer, test.initializer)

        _checks.check_elicit(
            test.model,
            test.parameters,
            test.targets,
            test.expert,
            test.trainer,
            test.optimizer,
            test.network,
            test.initializer,
            test.meta_settings,
        )

        # only if checks pass update variables of Elicit
        for i, key in enumerate(kwargs):
            setattr(self, key, kwargs[key])
            # reset results
            if hasattr(self, "results"):
                delattr(self, "results")
            self.temp_results = list()
            self.temp_history = list()
            if i == 0:
                # inform user about reset of results
                print("INFO: Results have been reset.")
        # a kwarg initializer=None must not replace the CMA-ES default
        self.initializer = test.initializer

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
        globals()["SEED"] = seed

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

        # initialization of hyperparameter
        (init_prior_model, loss_list, init_matrix) = initialization.init_prior(
            expert_elicits,
            self.initializer,
            self.parameters,
            self.trainer,
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
        if self.optimizer["optimizer"] == cmaes.CMAES:
            fit_method = cmaes.cma_training
            # the search starts where the initialization stopped. If the
            # initialization only read the box, the box also sets the first
            # step size of each coordinate.
            extra["default_sigma0"] = cmaes.box_step_size(
                self.initializer, expert_elicits, self.parameters
            )
        else:
            fit_method = optimization.sgd_training

        history, results = fit_method(
            expert_elicits,
            init_prior_model,
            self.trainer,
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
