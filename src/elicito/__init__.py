"""
A Python package for learning prior distributions based on expert knowledge
"""

import importlib.metadata
from typing import Any, Optional

import joblib
import tensorflow as tf
import tensorflow_probability as tfp  # type: ignore

from elicito import (
    initialization,
    losses,
    networks,
    optimization,
    plots,
    simulations,
    targets,
    types,
    utils,
)
from elicito.elicit import (
    expert,
    hyper,
    initializer,
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
    NFDict,
    Parallel,
    Parameter,
    SaveHist,
    SaveResults,
    Target,
    Trainer,
)

tfd = tfp.distributions

__version__ = importlib.metadata.version("elicito")

__all__ = [
    "Elicit",
    "expert",
    "hyper",
    "initialization",
    "initializer",
    "losses",
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
]

# global variable (gets overwritten by user-defined
# seed in Elicit object)
SEED = 0


class Elicit:
    """
    Configure the elicitation method
    """

    def __init__(  # noqa: PLR0912, PLR0913, PLR0915
        self,
        model: dict[str, Any],
        parameters: list[Parameter],
        targets: list[Target],
        expert: ExpertDict,
        trainer: Trainer,
        optimizer: dict[str, Any],
        network: Optional[NFDict] = None,
        initializer: Optional[Initializer] = None,
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
            Only required for ``parametric_prior`` method.

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
        # check expert data
        expected_dict = utils.get_expert_datformat(targets)
        try:
            expert["ground_truth"]
        except KeyError:
            # input expert data: ensure data has expected format
            if list(expert["data"].keys()) != list(expected_dict.keys()):
                msg = (
                    "[section: expert] Provided expert data is not in the "
                    + "correct format. Please use "
                    + "el.utils.get_expert_datformat to check expected format.",
                )
                raise AssertionError(msg)

        else:
            # oracle: ensure ground truth has same dim as number of model param
            expected_params = [param["name"] for param in parameters]
            num_params = 0
            for k in expert["ground_truth"]:
                # type list can result in cases where a tfd.Sequential/
                # Jointdistribution is used
                if type(expert["ground_truth"][k].sample(1)) is list:
                    num_params += sum(
                        [
                            param.shape[-1]
                            for i, param in enumerate(
                                expert["ground_truth"][k].sample(1)
                            )
                        ]
                    )
                else:
                    num_params += expert["ground_truth"][k].sample(1).shape[-1]

            if len(expected_params) != num_params:
                msg = (
                    "[section: expert] Dimensionality of ground truth in "  # type: ignore
                    + "'expert' is not the same  as number of model "
                    + f"parameters.Got {num_params=}, expected "
                    + f"{len(expected_params)}."
                )
                raise AssertionError(msg)

        # check that network architecture is provided when method is deep prior
        # and initializer is none
        if trainer["method"] == "deep_prior":
            if network is None:
                msg = (
                    "[section network] If method is 'deep prior', "
                    + " the section 'network' can't be None.",
                )
                raise ValueError(msg)

            if initializer is not None:
                msg = (
                    "[section initializer] For method 'deep_prior' the "
                    + "'initializer' is not used and should be set to None.",
                )
                raise ValueError(msg)

            if network["network_specs"]["num_params"] != len(parameters):
                msg = (
                    "[section network] The number of model parameters as "
                    + "specified in the parameters section, must match the "
                    + "number of parameters specified in the network (see "
                    + "network_specs['num_params'] argument).\n"
                    + f"Expected {len(parameters)} but got "
                    + f"{network['network_specs']['num_params']}",
                )
                raise ValueError(msg)

            if network["base_distribution"].__class__ != networks.BaseNormal:
                msg = (
                    "[network] Currently only the standard normal distribution "
                    + "is implemented as base distribution. "
                    + "See GitHub issue #35.",
                )
                raise NotImplementedError(msg)

        # check that initializer is provided when method=parametric prior
        # and network is none
        if trainer["method"] == "parametric_prior":
            if initializer is None:
                msg = (
                    "[section initializer] If method is 'parametric_prior', "
                    + " the section 'initializer' can't be None.",
                )
                raise ValueError(msg)

            if network is not None:
                msg = (
                    "[section network] If method is 'parametric prior' "
                    + "the 'network' is not used and should be set to None.",
                )
                raise ValueError(msg)

            # check that hyperparameter names are not redundant
            hyp_names = []
            hyp_shared = []
            for i in range(len(parameters)):
                if parameters[i]["hyperparams"] is None:
                    msg = (
                        "When using method='parametric_prior', the argument "
                        + "'hyperparams' of el.parameter "
                        + "cannot be None.",
                    )
                    raise ValueError(msg)

                hyp_names.append(
                    [
                        parameters[i]["hyperparams"][key]["name"]  # type: ignore
                        for key in parameters[i]["hyperparams"].keys()  # type: ignore
                    ]
                )
                hyp_shared.append(
                    [
                        parameters[i]["hyperparams"][key]["shared"]  # type: ignore
                        for key in parameters[i]["hyperparams"].keys()  # type: ignore
                    ]
                )
            # flatten nested list
            hyp_names_flat = sum(hyp_names, [])  # noqa: RUF017
            hyp_shared_flat = sum(hyp_shared, [])  # noqa: RUF017

            if initializer["method"] is None:
                for k in initializer["hyperparams"]:  # type: ignore
                    if k not in hyp_names_flat:
                        msg = (
                            f"[initializer] Hyperparameter name '{k}' doesn't "
                            + "match any name specified in the parameters "
                            + "section. Have you misspelled the name?",
                        )
                        raise ValueError(msg)

            seen = []
            duplicate = []
            share = []
            for n, s in zip(hyp_names_flat, hyp_shared_flat):
                if n not in seen:
                    seen.append(n)
                elif s:
                    share.append(n)
                else:
                    duplicate.append(n)

            if len(duplicate) != 0:
                msg = (
                    "[parameters] The following hyperparameter have the same "
                    + f"name but are not shared: {duplicate}. \n"
                    + "Have you forgot to set shared=True?",
                )
                raise ValueError(msg)

        self.model = model
        self.parameters = parameters
        self.targets = targets
        self.expert = expert
        self.trainer = trainer
        self.optimizer = optimizer
        self.network = network
        self.initializer = initializer

        self.history: list[dict[str, Any]] = []
        self.results: list[dict[str, Any]] = []

        # overwrite global seed
        globals()["SEED"] = self.trainer["seed"]

        # set seed
        tf.random.set_seed(SEED)

        # add seed information into model attribute
        # (required for discrete likelihood)
        # self.model["seed"] = self.trainer["seed"]

    def fit(
        self,
        save_history: SaveHist = utils.save_history(),
        save_results: SaveResults = utils.save_results(),
        overwrite: bool = False,
        parallel: Optional[Parallel] = None,
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

        save_history
            Exclude or include sub-results in the final result file.
            See [`save_history`][elicito.utils.save_history].
            In the ``history`` object are all results that are saved across epochs.
            TODO add link to notebook in docs

        save_results
            Exclude or include sub-results in the final result file.
            See [`save_results`][elicito.utils.save_results]
            In the ``results`` object are all results that are saved for the last
            epoch only.
            TODO add link to notebook in docs

        parallel
            specify parallelization settings if multiple trainings should run
            in parallel. See [`parallel`][elicito.utils.parallel].

        Examples
        --------
        >>> eliobj.fit()  # doctest: +SKIP

        >>> eliobj.fit(overwrite=True,  # doctest: +SKIP
        >>>            save_history=el.utils.save_history(  # doctest: +SKIP
        >>>                loss_component=False  # doctest: +SKIP
        >>>                )  # doctest: +SKIP
        >>>            )  # doctest: +SKIP

        >>> eliobj.fit(parallel=el.utils.parallel(runs=4))  # doctest: +SKIP

        """
        # set seed
        tf.random.set_seed(self.trainer["seed"])

        # check whether elicit object is already fitted
        refit = True
        if len(self.history) != 0 and not overwrite:
            user_answ = input(
                "eliobj is already fitted."
                + " Do you want to fit it again and overwrite the results?"
                + " Press 'n' to stop process and 'y' to continue fitting."
            )

            while user_answ not in ["n", "y"]:
                user_answ = input(
                    "Please press either 'y' for fitting or 'n'"
                    + " for abording the process."
                )

            if user_answ == "n":
                refit = False
                print("Process aborded; eliobj is not re-fitted.")

        # run single time if no parallelization is required
        if (parallel is None) and (refit):
            results, history = self.workflow(self.trainer["seed"])
            # include seed information into results
            results["seed"] = self.trainer["seed"]
            # remove results that user wants to exclude from saving
            results_prep, history_prep = utils.clean_savings(
                history, results, save_history, save_results
            )
            # save results in list attribute
            self.history.append(history_prep)
            self.results.append(results_prep)
        # run multiple replications
        if (parallel is not None) and (refit):
            # create a list of seeds if not provided
            if parallel["seeds"] is None:
                # generate seeds
                seeds = [
                    int(s) for s in tfd.Uniform(0, 999999).sample(parallel["runs"])
                ]
            else:
                seeds = parallel["seeds"]

            # run training simultaneously for multiple seeds
            (*res,) = joblib.Parallel(n_jobs=parallel["cores"])(
                joblib.delayed(self.workflow)(seed) for seed in seeds
            )

            for i, seed in enumerate(seeds):
                self.results.append(res[i][0])
                self.history.append(res[i][1])
                self.results[i]["seed"] = seed

                self.results[i], self.history[i] = utils.clean_savings(
                    self.history[i], self.results[i], save_history, save_results
                )

    def save(
        self,
        name: Optional[str] = None,
        file: Optional[str] = None,
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
        # check that either name or file is specified
        if not (name is None) ^ (file is None):
            msg = (
                "Name and file cannot be both None or both specified. "
                + "Either one has to be None.",
            )
            raise AssertionError(msg)

        # add a saving path
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

        for key in kwargs:  # noqa: PLC0206
            setattr(self, key, kwargs[key])
            # reset results
            self.results = list()
            self.history = list()
            # inform user about reset of results
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
        globals()["SEED"] = seed

        self.trainer["seed_chain"] = seed
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
        (init_prior_model, loss_list, init_prior_obj, init_matrix) = (
            initialization.init_prior(
                expert_elicits,
                self.initializer,
                self.parameters,
                self.trainer,
                self.model,
                self.targets,
                self.network,
                self.expert,
                seed,
                self.trainer["progress"],
            )
        )

        # run dag with optimal set of initial values
        # save results in corresp. attributes
        history, results = optimization.sgd_training(
            expert_elicits,
            init_prior_model,
            self.trainer,
            self.optimizer,
            self.model,
            self.targets,
            self.parameters,
            seed,
            self.trainer["progress"],
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
            results["init_prior"] = init_prior_obj
            results["init_matrix"] = init_matrix

        return tuple((results, history))
