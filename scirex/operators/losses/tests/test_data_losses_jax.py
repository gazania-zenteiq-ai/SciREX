import jax.numpy as jnp
from scirex.operators.losses.data_losses_jax import LpLoss


def test_lp_loss_basic():
    loss_fn = LpLoss(d=2, p=2)

    pred = jnp.ones((1, 1, 2, 2))
    target = jnp.ones((1, 1, 2, 2))

    loss = loss_fn(pred, target)

    assert jnp.allclose(loss, 0.0)


def test_lp_loss_l1():
    """Test LpLoss with L1 (relative behavior)."""

    loss_fn = LpLoss(d=2, p=1)

    pred = jnp.array([[[[1.0, 2.0], [3.0, 4.0]]]])
    target = jnp.array([[[[1.0, 1.0], [1.0, 1.0]]]])

    loss = loss_fn(pred, target)

    # Just check valid output (since it's relative loss)
    assert loss > 0
    assert jnp.isfinite(loss)


def test_lp_loss_relative():
    """Test relative behavior explicitly."""

    loss_fn = LpLoss(d=2, p=2)

    pred = jnp.array([[[[2.0, 2.0], [2.0, 2.0]]]])
    target = jnp.array([[[[1.0, 1.0], [1.0, 1.0]]]])

    loss = loss_fn(pred, target)

    assert loss > 0
    assert jnp.isfinite(loss)


def test_lp_loss_zero():
    """Zero loss when pred == target."""

    loss_fn = LpLoss(d=2, p=2)

    pred = jnp.zeros((1, 1, 2, 2))
    target = jnp.zeros((1, 1, 2, 2))

    loss = loss_fn(pred, target)

    assert jnp.allclose(loss, 0.0)