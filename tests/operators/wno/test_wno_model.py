import jax
import jax.numpy as jnp

from scirex.operators.models.wno import WNO
from scirex.operators.models.wno2d import WNO2D


def test_wno_1d_is_jittable():
    model = WNO(
        hidden_channels=8,
        n_layers=2,
        level=2,
        size=32,
        out_channels=2,
        wavelet="db2",
        mode="symmetric",
        use_grid=True,
    )
    x = jax.random.normal(jax.random.PRNGKey(0), (2, 31, 3))

    params = model.init(jax.random.PRNGKey(1), x)
    y = jax.jit(lambda p, x_: model.apply(p, x_))(params, x)

    assert y.shape == (2, 31, 2)
    assert jnp.isfinite(y).all()


def test_wno_2d_is_jittable():
    model = WNO(
        hidden_channels=8,
        n_layers=2,
        level=2,
        size=(18, 14),
        out_channels=3,
        wavelet="db4",
        mode="reflect",
        use_grid=True,
    )
    x = jax.random.normal(jax.random.PRNGKey(2), (2, 17, 13, 4))

    params = model.init(jax.random.PRNGKey(3), x)
    y = jax.jit(lambda p, x_: model.apply(p, x_))(params, x)

    assert y.shape == (2, 17, 13, 3)
    assert jnp.isfinite(y).all()


def test_wno2d_wrapper_matches_generic_api_shape():
    model = WNO2D(
        width=8,
        depth=2,
        level=2,
        size=(18, 14),
        out_channels=1,
        wavelet="db4",
        mode="reflect",
    )
    x = jax.random.normal(jax.random.PRNGKey(4), (2, 17, 13, 1))

    params = model.init(jax.random.PRNGKey(5), x)
    y = model.apply(params, x)

    assert y.shape == (2, 17, 13, 1)
    assert jnp.isfinite(y).all()
