import jax.numpy as jnp
from scirex.operators.training.adamw_jax import AdamW


# ✅ simple dummy parameter class
class DummyParam:
    def __init__(self, value):
        self.value = value
        self.grad = None

    def __array__(self):
        return self.value


def test_adamw_init():
    optimizer = AdamW(lr=0.001, weight_decay=0.01)

    assert optimizer.learning_rate == 0.001
    assert optimizer.param_groups[0]["weight_decay"] == 0.01


def test_adamw_step():
    # create dummy params
    p = DummyParam(jnp.ones((2, 2)))
    p.grad = jnp.ones((2, 2))

    optimizer = AdamW(params=[p], lr=0.01)

    loss, updates = optimizer.step()

    assert updates is not None
    assert isinstance(updates, dict)


def test_adamw_with_galore():
    p = DummyParam(jnp.ones((4, 4)))
    p.grad = jnp.ones((4, 4))

    optimizer = AdamW(
        params=[p],
        lr=0.001,
        galore_params=[p],
        galore_rank=2,
        galore_update_proj_gap=10,
    )

    loss, updates = optimizer.step()

    assert updates is not None
    assert isinstance(updates, dict)