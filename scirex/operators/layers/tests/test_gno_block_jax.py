import jax
import jax.numpy as jnp
from scirex.operators.layers.gno_block_jax import GNOBlock


def create_dummy_neighbors(n_points):
    """Create valid neighbors format for IntegralTransform"""
    neighbors_index = jnp.arange(n_points).repeat(n_points)
    segment_ids = jnp.repeat(jnp.arange(n_points), n_points)
    counts = jnp.ones(n_points) * n_points

    return {
        "neighbors_index": neighbors_index,
        "segment_ids": segment_ids,
        "counts": counts,
    }


def test_gno_block_integration():
    model = GNOBlock(
        in_channels=3,
        out_channels=3,
        coord_dim=3,
        radius=0.5,
        use_open3d_neighbor_search=False
    )

    key = jax.random.PRNGKey(0)

    n_points = 6

    y = jax.random.uniform(key, (n_points, 3))
    x = jax.random.uniform(key, (n_points, 3))
    f_y = jax.random.uniform(key, (1, n_points, 3))

    neighbors = create_dummy_neighbors(n_points)

    params = model.init(key, y, x, f_y, neighbors=neighbors)
    output = model.apply(params, y, x, f_y, neighbors=neighbors)

    assert output is not None
    assert jnp.isfinite(output).all()


def test_gno_block_integration_no_fy():
    model = GNOBlock(
        in_channels=3,
        out_channels=4,
        coord_dim=3,
        radius=0.5,
        transform_type="linear_kernelonly",
        use_open3d_neighbor_search=False
    )

    key = jax.random.PRNGKey(1)

    n_points = 5

    y = jax.random.uniform(key, (n_points, 3))
    x = jax.random.uniform(key, (n_points, 3))

    neighbors = create_dummy_neighbors(n_points)

    params = model.init(key, y, x, None, neighbors=neighbors)
    output = model.apply(params, y, x, None, neighbors=neighbors)

    assert output is not None
    assert jnp.isfinite(output).all()