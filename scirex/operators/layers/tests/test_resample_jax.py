import pytest
import jax.numpy as jnp
from scirex.operators.layers.resample_jax import resample


def test_resample_upscale():
    """Test resample with upscaling."""
    x = jnp.ones((1, 3, 8, 8))
    output = resample(x, res_scale=2, axis=(2, 3))

    assert output.shape == (1, 3, 16, 16)
    assert jnp.isfinite(output).all()


def test_resample_downscale():
    """Test resample with downscaling."""
    x = jnp.ones((1, 2, 16, 16))
    output = resample(x, res_scale=0.5, axis=(2, 3))

    assert output.shape == (1, 2, 8, 8)
    assert jnp.isfinite(output).all()


def test_resample_no_change():
    """Test resample with scale_factor=1."""
    x = jnp.ones((1, 4, 10, 10))
    output = resample(x, res_scale=1, axis=(2, 3))

    assert output.shape == (1, 4, 10, 10)
    assert jnp.isfinite(output).all()