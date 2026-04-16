import jax
import jax.numpy as jnp
import pytest

from scirex.operators.models.wno import WNO

@pytest.fixture
def prng_key():
    return jax.random.PRNGKey(42)

def test_wno_1d(prng_key):
    x = jax.random.normal(prng_key, (2, 64, 3))
    model = WNO(
        hidden_channels=16,
        n_layers=2,
        level=2,
        size=64,
        out_channels=5,
        wavelet="db2",
    )
    vars = model.init(prng_key, x)
    out = model.apply(vars, x)
    assert out.shape == (2, 64, 5)

def test_wno_2d(prng_key):
    x = jax.random.normal(prng_key, (2, 32, 32, 2))
    model = WNO(
        hidden_channels=16,
        n_layers=2,
        level=2,
        size=(32, 32),
        out_channels=1,
        wavelet="db4",
    )
    vars = model.init(prng_key, x)
    out = model.apply(vars, x)
    assert out.shape == (2, 32, 32, 1)

def test_wno_3d(prng_key):
    x = jax.random.normal(prng_key, (1, 16, 16, 16, 2))
    model = WNO(
        hidden_channels=8,
        n_layers=1,
        level=1,
        size=(16, 16, 16),
        out_channels=3,
        wavelet="db2",
    )
    vars = model.init(prng_key, x)
    out = model.apply(vars, x)
    assert out.shape == (1, 16, 16, 16, 3)
