import jax
import jax.numpy as jnp

from scirex.operators.layers.wavelet_conv import WaveletConv


def test_wavelet_conv_1d_is_jittable_and_shape_stable():
    module = WaveletConv(
        in_channels=3,
        out_channels=5,
        level=2,
        size=32,
        wavelet="db2",
        mode="symmetric",
    )
    x = jnp.ones((2, 31, 3), dtype=jnp.float32)

    params = module.init(jax.random.PRNGKey(0), x)
    y = jax.jit(lambda p, x_: module.apply(p, x_))(params, x)

    assert y.shape == (2, 31, 5)
    assert jnp.isfinite(y).all()


def test_wavelet_conv_2d_supports_reflect_mode():
    module = WaveletConv(
        in_channels=2,
        out_channels=4,
        level=2,
        size=(18, 14),
        wavelet="db4",
        mode="reflect",
    )
    x = jax.random.normal(jax.random.PRNGKey(1), (3, 17, 13, 2))

    params = module.init(jax.random.PRNGKey(2), x)
    y = jax.jit(lambda p, x_: module.apply(p, x_))(params, x)

    assert y.shape == (3, 17, 13, 4)
    assert jnp.isfinite(y).all()


def test_wavelet_conv_rejects_unsupported_modes():
    module = WaveletConv(
        in_channels=1,
        out_channels=1,
        level=1,
        size=16,
        wavelet="haar",
        mode="periodization",
    )
    x = jnp.ones((1, 16, 1), dtype=jnp.float32)

    try:
        module.init(jax.random.PRNGKey(0), x)
    except ValueError as exc:
        assert "Unsupported wavelet boundary mode" in str(exc)
    else:
        raise AssertionError("Expected ValueError for unsupported jaxwt mode.")


def test_wavelet_conv_zero_weights_produce_zero_output():
    module = WaveletConv(
        in_channels=1,
        out_channels=1,
        level=2,
        size=16,
        wavelet="haar",
        mode="symmetric",
    )
    x = jax.random.normal(jax.random.PRNGKey(3), (2, 15, 1))

    params = module.init(jax.random.PRNGKey(4), x)
    zero_params = {"params": jax.tree_util.tree_map(jnp.zeros_like, params["params"])}
    y = module.apply(zero_params, x)

    assert y.shape == (2, 15, 1)
    assert jnp.allclose(y, 0.0)
