"""
Check user input of Elicit object
"""

from elicito import networks, utils


def check_elicit(  # noqa: PLR0913, PLR0912, PLR0915
    model,
    parameters,
    targets,
    expert,
    trainer,
    optimizer,
    network,
    initializer,
    meta_settings,
) -> None:
    """
    Check whether input values to Elicit() are valid
    """
    # check expert data
    expected_dict = utils.get_expert_datformat(targets)
    try:
        expert["ground_truth"]
    except KeyError:
        # input expert data: ensure data has expected format
        if list(expert["data"].keys()) != list(expected_dict.keys()):
            msg = (
                "Provided expert data is not in the "
                "correct format. Please use "
                "el.utils.get_expert_datformat to check expected format."
            )
            raise AssertionError(msg)

    else:
        # oracle: ensure ground truth has same dim as number of model param
        expected_params = [param["name"] for param in parameters]
        num_params = 0
        if expert["ground_truth"] is None:
            pass
        else:
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
                "Dimensionality of ground truth in "  # type: ignore
                "'expert' is not the same  as number of model "
                f"parameters. Got {num_params=}, expected "
                f"{len(expected_params)}."
            )
        raise AssertionError(msg)

    # check that network architecture is provided when method is deep prior
    # and initializer is none
    if trainer["method"] == "deep_prior":
        if network is None:
            msg = "If method is 'deep prior', " " the section 'network' can't be None."
            raise ValueError(msg)

        if initializer is not None:
            msg = (
                "For method 'deep_prior' the "
                "'initializer' is not used and should be set to None."
            )
            raise ValueError(msg)

        if network["network_specs"]["num_params"] != len(parameters):
            msg = (
                "The number of model parameters as "
                "specified in the parameters section, must match the "
                "number of parameters specified in the network."
                f"Expected {len(parameters)} but got "
                f"{network['network_specs']['num_params']}"
            )
            raise ValueError(msg)

        if network["base_distribution"].__class__ != networks.BaseNormal:
            msg = (
                "Currently only the standard normal distribution "
                "is implemented as base distribution. "
                "See GitHub issue #35."
            )
            raise NotImplementedError(msg)

    # check that initializer is provided when method=parametric prior
    # and network is none
    if trainer["method"] == "parametric_prior":
        if initializer is None:
            msg = (
                "If method is 'parametric_prior', "
                " the section 'initializer' can't be None."
            )
            raise ValueError(msg)

        if network is not None:
            msg = (
                "If method is 'parametric prior' "
                "the 'network' is not used and should be set to None."
            )
            raise ValueError(msg)

        # check that hyperparameter names are not redundant
        hyp_names = []
        hyp_shared = []
        for i in range(len(parameters)):
            if parameters[i]["hyperparams"] is None:
                msg = (
                    "When using method='parametric_prior', the argument "
                    "'hyperparams' of el.parameter "
                    "cannot be None."
                )
                raise ValueError(msg)

            hyp_names.append(
                [
                    parameters[i]["hyperparams"][key]["name"]
                    for key in parameters[i]["hyperparams"].keys()
                ]
            )
            hyp_shared.append(
                [
                    parameters[i]["hyperparams"][key]["shared"]
                    for key in parameters[i]["hyperparams"].keys()
                ]
            )
        # flatten nested list
        hyp_names_flat = sum(hyp_names, [])  # noqa: RUF017
        hyp_shared_flat = sum(hyp_shared, [])  # noqa: RUF017

        if initializer["method"] is None:
            for k in initializer["hyperparams"]:  # type: ignore
                if k not in hyp_names_flat:
                    msg = (
                        f"Hyperparameter name '{k}' doesn't "
                        "match any name specified in the parameters "
                        "section. Have you misspelled the name?"
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
                "The following hyperparameter have the same "
                f"name but are not shared: {duplicate}. \n"
                "Have you forgot to set shared=True?"
            )
            raise ValueError(msg)
