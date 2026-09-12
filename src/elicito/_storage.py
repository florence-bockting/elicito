"""
Saving an eliobj to a file, and reading it back
"""

import os
import pickle
from typing import Any, Optional

import cloudpickle  # type: ignore


def save_as_pkl(obj: Any, save_dir: str) -> None:
    """
    Save file as pickle.

    Parameters
    ----------
    obj
        Variable that needs to be saved.

    save_dir
        Path indicating the file location.

    Examples
    --------
    >>> save_as_pkl(obj, "results/file.pkl")  # doctest: +SKIP

    """
    # if directory does not exist, create it
    os.makedirs(os.path.dirname(save_dir), exist_ok=True)
    # save obj to location as pickle
    serialized_obj = cloudpickle.dumps(obj)
    with open(save_dir, "wb") as file:
        pickle.dump(serialized_obj, file=file)


def save(
    eliobj: Any,
    name: Optional[str] = None,
    file: Optional[str] = None,
    overwrite: bool = False,
) -> None:
    """
    Save the eliobj as pickle.

    Parameters
    ----------
    eliobj
        Instance of the :func:`elicit.elicit.Elicit` class.

    name
        Name of the saved .pkl file.
        File is saved as .results/{method}/{name}_{seed}.pkl

    file
        Path to file, including file name,
        e.g. file="res" (saved as res.pkl) or
        file="method1/res" (saved as method1/res.pkl)

    overwrite
        Whether to overwrite existing file.

    Raises
    ------
    FileExistsError
        The file exists and ``overwrite`` is ``False``.

    """
    # either name or file must be specified
    if (name is not None) and (file is None):
        if name.endswith(".pkl"):
            name = name.removesuffix(".pkl")
        # create saving path
        path = f"./results/{eliobj.trainer['method']}/{name}_{eliobj.trainer['seed']}"
    elif (file is not None) and (name is None):
        # postprocess file to avoid file.pkl.pkl
        if file.endswith(".pkl"):
            file = file.removesuffix(".pkl")
        path = "./" + file
    else:
        msg = (
            "Name and file cannot be both None or both specified. "
            "Either one has to be None."
        )
        raise AssertionError(msg)

    if os.path.isfile(path + ".pkl") and not overwrite:
        msg = (
            f"The file '{path}.pkl' already exists. "
            "Use overwrite=True to replace it."
        )
        raise FileExistsError(msg)

    storage = dict()
    # user inputs
    storage["model"] = eliobj.model
    storage["parameters"] = eliobj.parameters
    storage["targets"] = eliobj.targets
    storage["expert"] = eliobj.expert
    storage["optimizer"] = eliobj.optimizer
    storage["trainer"] = eliobj.trainer
    storage["initializer"] = eliobj.initializer
    storage["network"] = eliobj.network
    # results
    if hasattr(eliobj, "results"):
        storage["results"] = eliobj.results
    else:
        storage["temp_results"] = []
        storage["temp_history"] = []

    save_as_pkl(storage, path + ".pkl")

    print(f"saved in: {path}.pkl")


def read_storage(file: str) -> dict[str, Any]:
    """
    Read the dictionary that [`save`][elicito._storage.save] stores

    Parameters
    ----------
    file
        path where ``eliobj`` object is saved.

    Returns
    -------
    storage :
        user inputs of the ``eliobj``, and its results if it was fitted.

    """
    with open(file, "rb") as f:
        obj_pickled = pickle.load(f)  # noqa: S301
    storage: dict[str, Any] = pickle.loads(obj_pickled)  # noqa: S301
    return storage
