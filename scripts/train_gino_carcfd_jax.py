from scirex.operators.data.datasets import get_dataset
from configs.gino_carcfd_config import Default
from scirex.operators.data.adapters import gino_batch_adapter
import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState
import optax

from scirex.operators.training.step_fns import train_step


# dummy model
def model_fn(x):
    return jnp.sum(x, axis=1)  # simple output

def main():
    config = Default()
    data_module = get_dataset("car_cfd", config)

    sample = data_module.train_data[0]
    batch = gino_batch_adapter(sample)

    # convert to jax arrays
    batch = {
        "x": jnp.array(batch["x"]),
        "y": jnp.array(batch["y"]),
    }

    # initialize params (dummy)
    params = {}

    # optimizer
    tx = optax.adam(1e-3)

    # create train state
    state = TrainState.create(
        apply_fn=lambda params, x: model_fn(x),
        params=params,
        tx=tx
    )

    # loss function
    def loss_fn(preds, y):
        return jnp.mean((preds - y) ** 2)

    # run one step
    state, metrics = train_step(state, batch, loss_fn)

    print("Loss:", metrics["loss"])


if __name__ == "__main__":
    main()