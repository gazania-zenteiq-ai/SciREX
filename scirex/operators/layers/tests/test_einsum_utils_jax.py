import pytest
import jax.numpy as jnp
from scirex.operators.layers.einsum_utils_jax import einsum_complexhalf


def test_einsum_complexhalf_basic():
    """Test einsum_complexhalf with basic operation."""
    a = jnp.array([1.0 + 1j, 2.0 + 2j])
    b = jnp.array([1.0 + 1j, 2.0 + 2j])
    output = einsum_complexhalf("i,i->", a, b)

    assert output.shape == ()
    assert jnp.isfinite(output).all()


def test_einsum_complexhalf_matrix():
    """Test einsum_complexhalf with matrix multiplication."""
    a = jnp.ones((2, 2), dtype=jnp.complex64)
    b = jnp.ones((2, 2), dtype=jnp.complex64)
    output = einsum_complexhalf("ij,jk->ik", a, b)

    assert output.shape == (2, 2)
    assert jnp.isfinite(output).all()


def test_einsum_complexhalf_real():
    """Test einsum_complexhalf with real inputs."""
    a = jnp.array([1.0, 2.0])
    b = jnp.array([3.0, 4.0])
    output = einsum_complexhalf("i,i->", a, b)

    assert output.shape == ()
    assert jnp.allclose(output, 11.0)
    assert jnp.isfinite(output).all()