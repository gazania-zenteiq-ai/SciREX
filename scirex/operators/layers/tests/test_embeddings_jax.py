import pytest
import jax
import jax.numpy as jnp
from scirex.operators.layers.embeddings_jax import SinusoidalEmbedding


def test_sinusoidal_embedding_transformer():
    model = SinusoidalEmbedding(
        in_channels=3,
        num_frequencies=16,
        embedding_type="transformer",
        max_positions=10000
    )

    x = jnp.ones((5, 3))
    output = model(x)

    expected_dim = model.in_channels * model.num_frequencies * 2

    assert output.shape == (5, expected_dim)
    assert jnp.isfinite(output).all()

def test_sinusoidal_embedding_nerf():
    """Test SinusoidalEmbedding with nerf type."""
    model = SinusoidalEmbedding(
        in_channels=2,
        num_frequencies=8,
        embedding_type="nerf"
    )

    x = jnp.ones((3, 2))
    output = model(x)

    # For nerf, out_channels = num_frequencies * 2 * in_channels
    assert output.shape == (3, 32)
    assert jnp.isfinite(output).all()


def test_sinusoidal_embedding_different_inputs():
    model = SinusoidalEmbedding(
        in_channels=4,
        num_frequencies=10,
        embedding_type="transformer"
    )

    x = jnp.ones((1, 4))
    output = model(x)

    expected_dim = model.in_channels * model.num_frequencies * 2

    assert output.shape == (1, expected_dim)
    assert jnp.isfinite(output).all()