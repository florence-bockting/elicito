import numpy as np

from elicito.parameters._base import numpy_seed
from elicito.parameters.networks import InvertibleNetwork


def test_two_flows_from_one_seed_permute_alike():
    """The fixed permutations become a function of the seed."""

    def build():
        with numpy_seed(123):
            return InvertibleNetwork(num_params=3, num_coupling_layers=2)

    first, second = build(), build()
    for l1, l2 in zip(first.coupling_layers, second.coupling_layers):
        assert np.array_equal(
            l1.permutation.permutation.numpy(), l2.permutation.permutation.numpy()
        )


def test_numpy_seed_restores_the_generator():
    """The generator of the caller survives the block."""
    np.random.seed(7)  # noqa: NPY002
    expected = np.random.permutation(5)  # noqa: NPY002
    np.random.seed(7)  # noqa: NPY002
    with numpy_seed(123):
        np.random.permutation(5)  # noqa: NPY002
    assert np.array_equal(np.random.permutation(5), expected)  # noqa: NPY002
