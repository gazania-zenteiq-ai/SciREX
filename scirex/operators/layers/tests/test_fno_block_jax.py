import pytest
import jax
import jax.numpy as jnp
from scirex.operators.layers.fno_block_jax import FNOBlocks


def test_fno_blocks_forward():
    """Test forward pass of FNOBlocks with dummy input."""
    # Initialize model with dummy parameters
    model = FNOBlocks(
        n_modes=((16, 16)),
        in_channels=64,
        out_channels=64,
        n_layers=2
    )

    # Create dummy input: (batch, in_channels, *n_modes)
    x = jnp.ones((1, 64, 16, 16))

    # Initialize parameters
    key = jax.random.PRNGKey(42)
    params = model.init(key, x)

    # Forward pass
    output = model.apply(params, x)

    # Assert output shape
    assert output.shape == (1, 64, 16, 16)

    # Assert output is finite
    assert jnp.isfinite(output).all()

    # Assert no NaN values
    assert not jnp.isnan(output).any()


def test_fno_blocks_multiple_layers():
    """Test FNOBlocks with multiple layers."""
    model = FNOBlocks(
        n_modes=((8, 8)),
        in_channels=32,
        out_channels=32,
        n_layers=4
    )

    x = jnp.ones((1, 32, 8, 8))
    key = jax.random.PRNGKey(0)
    params = model.init(key, x)
    output = model.apply(params, x)

    assert output.shape == (1, 32, 8, 8)
    assert jnp.isfinite(output).all()


def test_fno_blocks_different_channels():
    """Test FNOBlocks with different in/out channels."""
    model = FNOBlocks(
        n_modes=((4, 4)),
        in_channels=16,
        out_channels=16,
        n_layers=1
    )

    x = jnp.ones((1, 16, 4, 4))
    key = jax.random.PRNGKey(1)
    params = model.init(key, x)
    output = model.apply(params, x)

    assert output.shape == (1, 16, 4, 4)
    assert jnp.isfinite(output).all()