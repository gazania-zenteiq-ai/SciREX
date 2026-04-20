import pytest
import jax
import jax.numpy as jnp
from scirex.operators.layers.neighbor_search_jax import NeighborSearch


def test_neighbor_search_forward():
    """Test forward pass of NeighborSearch with dummy input."""
    # Initialize model
    model = NeighborSearch(use_open3d=False)

    # Create dummy inputs
    data = jnp.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=jnp.float32)
    queries = jnp.array([[0.5, 0.5, 0]], dtype=jnp.float32)
    radius = 1.0

    # Forward pass
    params = model.init(jax.random.PRNGKey(0), data, queries, radius)
    result = model.apply(params, data, queries, radius)

    # Assert keys exist
    assert 'neighbors_index' in result
    assert 'neighbors_row_splits' in result

    # Assert types
    assert isinstance(result['neighbors_index'], jnp.ndarray)
    assert isinstance(result['neighbors_row_splits'], jnp.ndarray)

    # Assert shapes
    assert result['neighbors_index'].ndim == 1
    assert result['neighbors_row_splits'].shape == (queries.shape[0] + 1,)


def test_neighbor_search_no_neighbors():
    """Test NeighborSearch with no neighbors found."""
    model = NeighborSearch(use_open3d=False)

    data = jnp.array([[0, 0, 0]], dtype=jnp.float32)
    queries = jnp.array([[10, 10, 10]], dtype=jnp.float32)
    radius = 1.0

    params = model.init(jax.random.PRNGKey(0), data, queries, radius)
    result = model.apply(params, data, queries, radius)

    assert 'neighbors_index' in result
    assert 'neighbors_row_splits' in result
    assert result['neighbors_row_splits'][0] == 0
    assert result['neighbors_row_splits'][1] == 0


def test_neighbor_search_return_norm():
    """Test NeighborSearch with return_norm=True."""
    model = NeighborSearch(use_open3d=False, return_norm=True)

    data = jnp.array([[0, 0, 0], [1, 0, 0]], dtype=jnp.float32)
    queries = jnp.array([[0.5, 0, 0]], dtype=jnp.float32)
    radius = 1.0

    params = model.init(jax.random.PRNGKey(0), data, queries, radius)
    result = model.apply(params, data, queries, radius)

    assert 'neighbors_index' in result
    assert 'neighbors_row_splits' in result
    assert 'weights' in result
    assert isinstance(result['weights'], jnp.ndarray)