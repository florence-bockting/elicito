import numpy as np
import pytest

from elicito.networks import (
    ActNorm,
    AffineCoupling,
    CouplingLayer,
    InvertibleNetwork,
    Orthogonal,
    Permutation,
    SplineCoupling,
)


@pytest.mark.parametrize("coupling_design", ["affine", "spline"])
@pytest.mark.parametrize("permutation", ["fixed", "learnable"])
@pytest.mark.parametrize("use_act_norm", [True, False])
@pytest.mark.parametrize("input_shape", ["2d", "3d"])
def test_coupling_layer(coupling_design, permutation, use_act_norm, input_shape):
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

    # Forward and inverse pass
    z, ldj = network(inp, None)
    z = z.numpy()
    inp_rec = network(z, None, inverse=True).numpy()

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


@pytest.mark.parametrize("input_shape", ["2d", "3d"])
@pytest.mark.parametrize("use_soft_flow", [True, False])
@pytest.mark.parametrize("permutation", ["learnable", "fixed"])
@pytest.mark.parametrize("coupling_design", ["affine", "spline", "interleaved"])
@pytest.mark.parametrize("num_coupling_layers", [2, 7])
def test_invertible_network(  # noqa: PLR0912
    input_shape, use_soft_flow, permutation, coupling_design, num_coupling_layers
):
    """Tests the ``InvertibleNetwork``

    core class using a couple of relevant configurations.

    """

    # Randomize units and input dim
    units = np.random.randint(low=2, high=32)  # noqa: NPY002
    input_dim = np.random.randint(low=2, high=32)  # noqa: NPY002

    # Create settings dictionaries
    if coupling_design in ["affine", "spline"]:
        coupling_settings = {
            "dense_args": dict(units=units, activation="elu"),
            "num_dense": 1,
        }
    else:
        coupling_settings = {
            "affine": dict(
                dense_args={"units": units, "activation": "selu"}, num_dense=1
            ),
            "spline": dict(
                dense_args={"units": units, "activation": "relu"}, bins=8, num_dense=1
            ),
        }

    # Create invertible network with test settings
    network = InvertibleNetwork(
        num_params=input_dim,
        num_coupling_layers=num_coupling_layers,
        use_soft_flow=use_soft_flow,
        permutation=permutation,
        coupling_design=coupling_design,
        coupling_settings=coupling_settings,
    )

    # Create randomized input and output conditions
    batch_size = np.random.randint(low=1, high=32)  # noqa: NPY002
    if input_shape == "2d":
        inp = np.random.normal(size=(batch_size, input_dim)).astype(np.float32)  # noqa: NPY002
    else:
        n_obs = np.random.randint(low=1, high=32)  # noqa: NPY002
        inp = np.random.normal(size=(batch_size, n_obs, input_dim)).astype(np.float32)  # noqa: NPY002

    # Forward and inverse pass
    z, ldj = network(inp, None)
    z = z.numpy()
    inp_rec = network(z, None, inverse=True).numpy()

    # Test attributes
    assert network.latent_dim == input_dim
    assert len(network.coupling_layers) == num_coupling_layers
    # Test layer attributes
    for idx, l in enumerate(network.coupling_layers):  # noqa: E741
        # Permutation
        if permutation == "fixed":
            assert isinstance(l.permutation, Permutation)
        elif permutation == "learnable":
            assert isinstance(l.permutation, Orthogonal)
        # Default ActNorm
        assert isinstance(l.act_norm, ActNorm)
        # Coupling type
        if coupling_design == "affine":
            assert isinstance(l.net1, AffineCoupling) and isinstance(
                l.net2, AffineCoupling
            )
        elif coupling_design == "spline":
            assert isinstance(l.net1, SplineCoupling) and isinstance(
                l.net2, SplineCoupling
            )
        elif coupling_design == "interleaved":
            if idx % 2 == 0:
                assert isinstance(l.net1, AffineCoupling) and isinstance(
                    l.net2, AffineCoupling
                )
            else:
                assert isinstance(l.net1, SplineCoupling) and isinstance(
                    l.net2, SplineCoupling
                )

    if use_soft_flow:
        assert network.soft_flow is True
    else:
        assert network.soft_flow is False
    # Test invertibility (in case no soft flow)
    if not use_soft_flow:
        assert np.allclose(inp, inp_rec, atol=1e-5)
    # Test shapes (bijectivity)
    assert z.shape == inp.shape
    assert z.shape[-1] == input_dim
    if input_shape == "2d":
        assert ldj.shape[0] == inp.shape[0]
    else:
        assert ldj.shape[0] == inp.shape[0] and ldj.shape[1] == inp.shape[1]
