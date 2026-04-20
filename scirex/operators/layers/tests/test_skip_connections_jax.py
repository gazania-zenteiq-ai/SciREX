import jax
import pytest
import jax.numpy as jnp
from scirex.operators.layers.skip_connections_jax import skip_connection


def test_skip_connection_identity():
    model = skip_connection(
        in_features=8,
        out_features=8,
        skip_type="identity"
    )

    x = jnp.ones((1, 8, 4, 4))
    key = jax.random.PRNGKey(0)

    params = model.init(key, x)
    output = model.apply(params, x)

    assert output.shape == x.shape


def test_skip_connection_linear():
    model = skip_connection(
        in_features=8,
        out_features=8,
        skip_type="linear"
    )

    x = jnp.ones((1, 8, 4, 4))
    key = jax.random.PRNGKey(1)

    params = model.init(key, x)
    output = model.apply(params, x)

    assert output.shape == x.shape


def test_skip_connection_soft_gating():
    model = skip_connection(
        in_features=8,
        out_features=8,
        skip_type="soft-gating"
    )

    x = jnp.ones((1, 8, 4, 4))
    key = jax.random.PRNGKey(2)

    params = model.init(key, x)
    output = model.apply(params, x)

    assert output.shape == x.shape