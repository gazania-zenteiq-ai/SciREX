import pytest
import jax.numpy as jnp
from scirex.operators.layers.gno_weighting_functions_jax import dispatch_weighting_fn


def test_dispatch_weighting_fn_valid():
    fn = dispatch_weighting_fn("bump", sq_radius=0.1, scale=1.0)

    x = jnp.array([0.1, 0.2])
    out = fn(x)

    assert out.shape == x.shape
    assert jnp.isfinite(out).all()


def test_dispatch_weighting_fn_none():
    with pytest.raises(NotImplementedError):
        dispatch_weighting_fn(None, sq_radius=0.1, scale=1.0)


def test_dispatch_weighting_fn_invalid():
    with pytest.raises(NotImplementedError):
        dispatch_weighting_fn("invalid", sq_radius=0.1, scale=1.0)