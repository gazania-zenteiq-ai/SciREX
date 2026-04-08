from scirex.operators.models.gino_jax import GINO
import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState
import optax

from scirex.operators.training.step_fns import train_step
from scirex.operators.data.datasets import get_dataset
from configs.gino_carcfd_config import Default as Config
from scirex.operators.data.adapters import gino_batch_adapter


def main():
    # ✅ config
    config = Config()

    # ✅ dataset
    data_module = get_dataset("car_cfd", config)

    # 🔥 initialize model FIRST (needed for neighbors)
    model = GINO(
        in_channels=1,
        out_channels=1
    )

    # 🔥 VERY IMPORTANT: precompute neighbors
    print("Precomputing neighbors...")
    model.precompute_neighbors(data_module, split="train")
    print("Done!")

    # ✅ take one sample AFTER neighbors computed
    sample = data_module.train_data[0]

    # ✅ convert to GINO batch format
    batch = gino_batch_adapter(sample)

    # ✅ convert ALL to jax arrays
    batch = {k: jnp.array(v) if k not in ["neighbors_in", "neighbors_out"] else v
             for k, v in batch.items()}

    # 🔥 initialize parameters
    key = jax.random.PRNGKey(0)

    params = model.init(
        key,
        batch["input_geom"],
        batch["latent_queries"],
        batch["output_queries"],
        x=batch["x"],
        neighbors_in=batch["neighbors_in"],
        neighbors_out=batch["neighbors_out"],
    )

    # 🔥 optimizer
    tx = optax.adam(1e-3)

    # 🔥 create train state
    state = TrainState.create(
        apply_fn=lambda params, batch: model.apply(
            params,
            batch["input_geom"],
            batch["latent_queries"],
            batch["output_queries"],
            x=batch["x"],
            neighbors_in=batch["neighbors_in"],
            neighbors_out=batch["neighbors_out"],
        ),
        params=params,
        tx=tx
    )

    # 🔥 loss function
    def loss_fn(preds, y):
        preds = preds.squeeze()
        return jnp.mean((preds - y) ** 2)

    # 🔥 train step
    state, metrics = train_step(state, batch, loss_fn)

    print("GINO Loss:", metrics["loss"])


if __name__ == "__main__":
    main()