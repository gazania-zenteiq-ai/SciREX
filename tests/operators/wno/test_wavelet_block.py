import jax
import jax.numpy as jnp

from scirex.operators.layers.wavelet_block import WaveletBlock


def test_wavelet_block_projects_input_channels_and_is_jittable():
    module = WaveletBlock(
        hidden_channels=6,
        level=2,
        size=(18, 14),
        wavelet="db2",
        mode="symmetric",
        activation=None,
    )
    x = jax.random.normal(jax.random.PRNGKey(0), (2, 17, 13, 3))

    params = module.init(jax.random.PRNGKey(1), x)
    y = jax.jit(lambda p, x_: module.apply(p, x_))(params, x)

    assert y.shape == (2, 17, 13, 6)
    assert jnp.isfinite(y).all()

def test_wavelet_block_propagates_mode_for_2d_inputs():
    module = WaveletBlock(
        hidden_channels=4,
        level=2,
        size=(18, 14),
        wavelet="db4",
        mode="reflect",
        activation=None,
    )
    x = jax.random.normal(jax.random.PRNGKey(2), (3, 17, 13, 4))

    params = module.init(jax.random.PRNGKey(3), x)
    y = module.apply(params, x)

    assert y.shape == (3, 17, 13, 4)
    assert jnp.isfinite(y).all()
