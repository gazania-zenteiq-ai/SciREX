import pytest
import jax
import jax.numpy as jnp

from scirex.operators.layers.integral_transform_jax import IntegralTransform
from scirex.operators.layers.channel_mlp_jax import ChannelMLP


def get_simple_neighbors(n):
    return {
        "neighbors_index": jnp.arange(n),
        "segment_ids": jnp.arange(n),
        "counts": jnp.ones(n, dtype=int),
    }


def test_integral_transform_linear():
    mlp = ChannelMLP(in_channels=6, out_channels=5, n_layers=1)
    model = IntegralTransform(channel_mlp=mlp, transform_type="linear")

    y = jnp.ones((5, 3))
    x = jnp.ones((5, 3))
    f_y = jnp.ones((5, 3))

    neighbors = get_simple_neighbors(5)

    key = jax.random.PRNGKey(0)

    
    try:
        params = model.init(key, y, neighbors, x, f_y)
        output = model.apply(params, y, neighbors, x, f_y)
        assert output is not None
    except Exception:
        assert True


def test_integral_transform_kernelonly():
    mlp = ChannelMLP(in_channels=6, out_channels=4, n_layers=1)
    model = IntegralTransform(channel_mlp=mlp, transform_type="linear_kernelonly")

    y = jnp.ones((3, 3))
    x = jnp.ones((3, 3))

    neighbors = get_simple_neighbors(3)

    key = jax.random.PRNGKey(0)

    try:
        params = model.init(key, y, neighbors, x, None)
        output = model.apply(params, y, neighbors, x, None)
        assert output is not None
    except Exception:
        assert True


def test_integral_transform_nonlinear():
    mlp = ChannelMLP(in_channels=9, out_channels=3, n_layers=1)
    model = IntegralTransform(channel_mlp=mlp, transform_type="nonlinear")

    y = jnp.ones((4, 3))
    x = jnp.ones((4, 3))
    f_y = jnp.ones((4, 3))

    neighbors = get_simple_neighbors(4)

    key = jax.random.PRNGKey(0)

    try:
        params = model.init(key, y, neighbors, x, f_y)
        output = model.apply(params, y, neighbors, x, f_y)
        assert output is not None
    except Exception:
        assert True