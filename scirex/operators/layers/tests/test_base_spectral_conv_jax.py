import jax.numpy as jnp
from scirex.operators.layers.base_spectral_conv_jax import BaseSpectralConv


def test_base_spectral_conv_transform():
    model = BaseSpectralConv()

    x = jnp.ones((1, 3, 16, 16))

    output = model.transform(x)

    assert output.shape == x.shape
    assert (output == x).all()