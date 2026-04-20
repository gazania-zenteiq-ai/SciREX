import pytest
import jax.numpy as jnp
from scirex.operators.layers.padding_jax import DomainPadding


def test_domain_padding_forward():
    """Test forward pass of DomainPadding with dummy input."""
    # Initialize model
    model = DomainPadding(padding=0.125)

    # Create dummy input: (batch, channels, H, W)
    x = jnp.ones((1, 16, 16, 3))

    # Forward pass
    output = model(x)

    # Assert output shape (padded)
    # Assuming padding adds to each side
    expected_H = int(16 * (1 + 2 * 0.125))
    expected_W = int(16 * (1 + 2 * 0.125))
    assert output.shape == (1, expected_H, expected_W, 3)

    # Assert output is finite
    assert jnp.isfinite(output).all()


def test_domain_padding_no_padding():
    """Test DomainPadding with zero padding."""
    model = DomainPadding(padding=0.0)

    x = jnp.ones((1, 2, 8, 8))
    output = model(x)

    assert output.shape == (1, 2, 8, 8)
    assert jnp.isfinite(output).all()


def test_domain_padding_different_dims():
    """Test DomainPadding with 3D input."""
    model = DomainPadding(padding=0.1)

    x = jnp.ones((1, 1, 4, 4, 4))
    output = model(x)

    expected = int(4 * (1 + 2 * 0.1))
    assert output.shape == (1, 1, expected, expected, expected)
    assert jnp.isfinite(output).all()