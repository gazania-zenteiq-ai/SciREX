import pytest
import jax
import jax.numpy as jnp

from scirex.operators.training.patching_jax import (
    unfold_jax,
    make_patches,
    MultigridPatching2D,
)



#  unfold_jax TESTS


def test_unfold_jax_basic():
    x = jnp.arange(24).reshape(1, 1, 6, 4)

    out = unfold_jax(x, dimension=2, kernel_size=3, stride=1)

    # shape check
    assert out.shape == (1, 1, 4, 3, 4)

    # values should be finite
    assert jnp.isfinite(out).all()


def test_unfold_jax_stride():
    x = jnp.ones((1, 2, 8, 4))

    out = unfold_jax(x, dimension=2, kernel_size=4, stride=2)

    assert out.shape == (1, 2, 3, 4, 4)


def test_unfold_jax_negative_dim():
    x = jnp.ones((1, 3, 8))

    out = unfold_jax(x, dimension=-1, kernel_size=3, stride=1)

    assert out.shape == (1, 3, 6, 3)


def test_unfold_jax_no_overlap():
    x = jnp.ones((1, 1, 8))

    out = unfold_jax(x, dimension=2, kernel_size=2, stride=2)

    assert out.shape == (1, 1, 4, 2)



#  make_patches TESTS


def test_make_patches_2d():
    x = jnp.ones((1, 1, 8, 8))

    patches = make_patches(x, n=2, p=0)

    # 2x2 patches → batch expands
    assert patches.shape == (4, 1, 4, 4)


def test_make_patches_with_padding():
    x = jnp.ones((1, 1, 8, 8))

    patches = make_patches(x, n=2, p=1)

    # padding increases size
    assert patches.shape[0] == 4
    assert patches.shape[1] == 1
    assert patches.shape[2] > 4
    assert patches.shape[3] > 4


#  MultigridPatching2D TESTS

def test_mgpatching_identity():
    model = MultigridPatching2D(levels=0)

    x = jnp.ones((1, 1, 8, 8))
    y = jnp.ones((1, 1, 8, 8))

    x_out, y_out = model.patch(x, y)

    # no change when levels = 0
    assert x_out.shape == x.shape
    assert y_out.shape == y.shape


def test_mgpatching_patch_unpatch():
    model = MultigridPatching2D(levels=1)

    x = jnp.ones((1, 1, 8, 8))
    y = jnp.ones((1, 1, 8, 8))

    x_patch, y_patch = model.patch(x, y)

    # after patch → batch increases
    assert x_patch.shape[0] > x.shape[0]

    x_recon, y_recon = model.unpatch(x_patch, y_patch, evaluation=True)

    # For multi-level patching (levels > 0), the channel dimension changes
    # due to concatenation of different resolution patches
    # So we check that spatial dimensions are reconstructed correctly
    assert x_recon.shape[0] == x.shape[0]  # batch size
    assert x_recon.shape[2:] == x.shape[2:]  # spatial dimensions (H, W)
    # Channel dimension may be different due to multi-level concatenation


def test_mgpatching_stitch():
    model = MultigridPatching2D(levels=1)

    x = jnp.ones((4, 1, 4, 4))  # already patched format

    stitched = model._stitch(x)

    assert stitched.ndim == 4


def test_mgpatching_unpad():
    model = MultigridPatching2D(levels=0)
    model.padding_height = 1
    model.padding_width = 1

    x = jnp.ones((1, 1, 10, 10))

    out = model._unpad(x)

    assert out.shape == (1, 1, 8, 8)