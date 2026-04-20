import jax.numpy as jnp
from scirex.operators.layers.segment_csr_jax import segment_csr


def test_segment_csr_sum():
    """Test segment_csr with sum reduction."""

    # 4 edges, 2 segments
    src = jnp.array([[1.0], [2.0], [3.0], [4.0]])

    segment_ids = jnp.array([0, 0, 1, 1])
    counts = jnp.array([2, 2])

    output = segment_csr(src, segment_ids, counts, reduction="sum")

    expected = jnp.array([[3.0], [7.0]])

    assert output.shape == expected.shape
    assert jnp.allclose(output, expected)


def test_segment_csr_mean():
    """Test segment_csr with mean reduction."""

    src = jnp.array([[1.0], [2.0], [3.0], [4.0]])

    segment_ids = jnp.array([0, 0, 1, 1])
    counts = jnp.array([2, 2])

    output = segment_csr(src, segment_ids, counts, reduction="mean")

    expected = jnp.array([[1.5], [3.5]])

    assert output.shape == expected.shape
    assert jnp.allclose(output, expected)


def test_segment_csr_single_segment():
    """Test segment_csr with single segment."""

    src = jnp.array([[5.0], [6.0]])

    segment_ids = jnp.array([0, 0])
    counts = jnp.array([2])

    output = segment_csr(src, segment_ids, counts, reduction="sum")

    expected = jnp.array([[11.0]])

    assert output.shape == expected.shape
    assert jnp.allclose(output, expected)