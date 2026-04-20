import pytest
import jax
import jax.numpy as jnp
from scirex.operators.layers.spectral_convolution_jax import SpectralConv


def test_spectral_conv_forward():
    """Test forward pass of SpectralConv with dummy input."""
    # Initialize model with dummy parameters
    model = SpectralConv(
        n_modes=((8, 8, 8)),
        in_channels=3,
        out_channels=64,
        factorization=None,
        rank=1.0
    )

    # Create dummy input: (batch, in_channels, *n_modes)
    x = jnp.ones((1, 3, 8, 8, 8))

    # Initialize parameters
    key = jax.random.PRNGKey(42)
    params = model.init(key, x)

    # Forward pass
    output = model.apply(params, x)

    # Assert output shape
    assert output.shape == (1, 64, 8, 8, 8)

    # Assert output is finite
    assert jnp.isfinite(output).all()

    # Assert no NaN values
    assert not jnp.isnan(output).any()


def test_spectral_conv_different_modes():
    """Test SpectralConv with different mode configurations."""
    model = SpectralConv(
        n_modes=((4, 4)),
        in_channels=2,
        out_channels=32
    )

    x = jnp.ones((1, 2, 4, 4))
    key = jax.random.PRNGKey(0)
    params = model.init(key, x)
    output = model.apply(params, x)

    assert output.shape == (1, 32, 4, 4)
    assert jnp.isfinite(output).all()


def test_spectral_conv_factorized():
    """Test SpectralConv with factorization."""
    model = SpectralConv(
        n_modes=((8, 8)),
        in_channels=4,
        out_channels=16,
        factorization="tucker",
        rank=0.5
    )

    x = jnp.ones((1, 4, 8, 8))
    key = jax.random.PRNGKey(1)
    params = model.init(key, x)
    output = model.apply(params, x)

    assert output.shape == (1, 16, 8, 8)
    assert jnp.isfinite(output).all()