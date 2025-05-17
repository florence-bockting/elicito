import numpy as np
import pytest

from elicito.networks import (
    AffineCoupling,
    CouplingLayer,
    Orthogonal,
    Permutation,
    SplineCoupling,
)


@pytest.mark.parametrize("condition", [False])
@pytest.mark.parametrize("coupling_design", ["affine", "spline"])
@pytest.mark.parametrize("permutation", ["fixed", "learnable"])
@pytest.mark.parametrize("use_act_norm", [True, False])
@pytest.mark.parametrize("input_shape", ["2d", "3d"])
def test_coupling_layer(  # noqa: PLR0912
    condition, coupling_design, permutation, use_act_norm, input_shape
):
    """Tests the ``CouplingLayer`` instance with various configurations."""

    # Randomize units and input dim
    units = np.random.randint(low=2, high=32)  # noqa: NPY002
    input_dim = np.random.randint(low=2, high=32)  # noqa: NPY002

    # Create settings dictionaries and network
    if coupling_design == "affine":
        coupling_settings = {
            "dense_args": dict(units=units, activation="elu"),
            "num_dense": 1,
        }
    else:
        coupling_settings = {
            "dense_args": dict(units=units, activation="elu"),
            "num_dense": 1,
            "bins": 8,
        }
    settings = {
        "latent_dim": input_dim,
        "coupling_settings": coupling_settings,
        "permutation": permutation,
        "use_act_norm": use_act_norm,
        "coupling_design": coupling_design,
    }

    network = CouplingLayer(**settings)

    # Create randomized input and output conditions
    batch_size = np.random.randint(low=1, high=32)  # noqa: NPY002
    if input_shape == "2d":
        inp = np.random.normal(size=(batch_size, input_dim)).astype(np.float32)  # noqa: NPY002
    else:
        n_obs = np.random.randint(low=1, high=32)  # noqa: NPY002
        inp = np.random.normal(size=(batch_size, n_obs, input_dim)).astype(np.float32)  # noqa: NPY002
    if condition:
        condition_dim = np.random.randint(low=1, high=32)  # noqa: NPY002
        condition = np.random.normal(size=(batch_size, condition_dim)).astype(  # noqa: NPY002
            np.float32
        )
    else:
        condition = None

    # Forward and inverse pass
    z, ldj = network(inp, condition)
    z = z.numpy()
    inp_rec = network(z, condition, inverse=True).numpy()

    # Test attributes
    if permutation == "fixed":
        assert not network.permutation.trainable
        assert isinstance(network.permutation, Permutation)
    else:
        assert isinstance(network.permutation, Orthogonal)
        assert network.permutation.trainable
    if use_act_norm:
        assert network.act_norm is not None
    else:
        assert network.act_norm is None

    # Test coupling type
    if coupling_design == "affine":
        assert isinstance(network.net1, AffineCoupling) and isinstance(
            network.net2, AffineCoupling
        )
    elif coupling_design == "spline":
        assert isinstance(network.net1, SplineCoupling) and isinstance(
            network.net2, SplineCoupling
        )

    # Test invertibility
    assert np.allclose(inp, inp_rec, atol=1e-5)
    # Test shapes (bijectivity)
    assert z.shape == inp.shape
    if input_shape == "2d":
        assert ldj.shape[0] == inp.shape[0]
    else:
        assert ldj.shape[0] == inp.shape[0] and ldj.shape[1] == inp.shape[1]
