"""
Check user input of Elicit object
"""

from elicito import utils
from elicito.optimizers import cmaes
from elicito.parameters import methods


def check_elicit(  # type: ignore  # noqa: PLR0913
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
                # sample once; .sample() is not free and the shape is all we need
                ground_truth_sample = expert["ground_truth"][k].sample(1)
                # type list can result in cases where a tfd.Sequential/
                # Jointdistribution is used
                if type(ground_truth_sample) is list:
                    num_params += sum(
                        [param.shape[-1] for param in ground_truth_sample]
                    )
                else:
                    num_params += ground_truth_sample.shape[-1]

        if len(expected_params) != num_params:
            msg = (
                "Dimensionality of ground truth in "
                "'expert' is not the same  as number of model "
                f"parameters. Got {num_params=}, expected "
                f"{len(expected_params)}."
            )
            raise AssertionError(msg)

    # CMA-ES keeps a full covariance matrix of the search space. The weights
    # of a normalizing flow are thousands of dimensions, which it cannot search.
    if optimizer["optimizer"] == cmaes.CMAES and trainer["method"] == "deep_prior":
        msg = (
            f"optimizer='{cmaes.CMAES}' can only be used with "
            "method='parametric_prior'. The 'deep_prior' method has too many "
            "trainable variables for a derivative-free search. Use a "
            "tf.keras optimizer instead."
        )
        raise ValueError(msg)

    methods.get_method(trainer["method"]).check(parameters, network, initializer)
