import pytest
import jax.numpy as jnp
from scirex.operators.training.tensor_galore_projector_jax import TensorGaLoreProjector


def test_tensor_galore_projector_init():
    """Test TensorGaLoreProjector initialization."""
    projector = TensorGaLoreProjector(rank=5, update_proj_gap=10, scale=1.0)

    assert projector.rank == 5
    assert projector.update_proj_gap == 10
    assert projector.scale == 1.0


def test_tensor_galore_projector_project():
    """Test projection of gradients."""
    projector = TensorGaLoreProjector(rank=3)

    grad = jnp.ones((6, 6))
    projected = projector.project(grad, iter=0)

    assert projected.shape[0] == 3
    assert projected.shape[1] == 3
    assert jnp.isfinite(projected).all()


def test_tensor_galore_projector_update():
    """Test updating projection matrix."""
    projector = TensorGaLoreProjector(rank=2, update_proj_gap=1)

    grad = jnp.ones((4, 4))
    projector.project(grad, iter=0) # This should trigger update

    assert projector.proj_tensor is not None
    for factor in projector.proj_tensor:
        assert jnp.isfinite(factor).all()


def test_tensor_galore_projector_rank_percentage():
    """Test with rank as percentage."""
    projector = TensorGaLoreProjector(rank=0.5)  # 50%

    grad = jnp.ones((10, 10))
    projected = projector.project(grad, iter=0)

    assert projected.shape[0] == 2
    assert projected.shape[1] == 2
    assert jnp.isfinite(projected).all()