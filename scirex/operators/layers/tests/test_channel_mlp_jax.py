import pytest
import jax
import jax.numpy as jnp
import flax.linen as nn 
from scirex.operators.layers.channel_mlp_jax import ChannelMLP


def test_channel_mlp_forward():
    """Test forward pass of ChannelMLP with dummy input."""
    # Initialize model with dummy parameters
    model = ChannelMLP(
        in_channels=10,
        out_channels=5,
        hidden_channels=20,
        n_layers=2,
        non_linearity=nn.gelu
    )

    # Create dummy input: (batch, channels, H, W)
    x = jnp.ones((2, 10, 32, 32))

    # Initialize parameters
    key = jax.random.PRNGKey(42)
    params = model.init(key, x)

    # Forward pass
    output = model.apply(params, x)

    # Assert output shape
    assert output.shape == (2, 5, 32, 32)

    # Assert output is finite
    assert jnp.isfinite(output).all()

    # Assert no NaN values
    assert not jnp.isnan(output).any()


def test_channel_mlp_single_layer():
    """Test ChannelMLP with single layer."""
    model = ChannelMLP(in_channels=8, out_channels=8, n_layers=1)

    x = jnp.ones((1, 8, 16, 16))
    key = jax.random.PRNGKey(0)
    params = model.init(key, x)
    output = model.apply(params, x)

    assert output.shape == (1, 8, 16, 16)
    assert jnp.isfinite(output).all()


def test_channel_mlp_no_hidden():
    """Test ChannelMLP with no hidden channels specified."""
    model = ChannelMLP(in_channels=4, out_channels=2)

    x = jnp.ones((1, 4, 8, 8))
    key = jax.random.PRNGKey(1)
    params = model.init(key, x)
    output = model.apply(params, x)

    assert output.shape == (1, 2, 8, 8)
    assert jnp.isfinite(output).all()