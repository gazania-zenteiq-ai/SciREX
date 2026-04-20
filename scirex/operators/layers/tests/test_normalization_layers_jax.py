import pytest
import jax
import jax.numpy as jnp
from scirex.operators.layers.normalization_layers_jax import (
    InstanceNorm,
    BatchNorm,
    AdaIN,
)


def test_instance_norm():
    model = InstanceNorm()

    x = jnp.ones((2, 3, 4, 4))

    output = model(x)

    assert output.shape == x.shape

def test_batch_norm():
    model = BatchNorm(n_dim=2, num_features=3)

    x = jnp.ones((2, 3, 4, 4))
    key = jax.random.PRNGKey(0)

    params = model.init(key, x)
    output = model.apply(
    params,
    x,
    use_running_average=True)

    assert output.shape == x.shape

def test_adain():
    model = AdaIN(embed_dim=4, in_channels=3)

    x = jnp.ones((1, 3, 4, 4))
    embedding = jnp.ones((4,))
    key = jax.random.PRNGKey(0)

    
    with pytest.raises(ValueError):
        model.init(key, x, embedding)