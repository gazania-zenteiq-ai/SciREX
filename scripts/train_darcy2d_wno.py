#!/usr/bin/env python3
# Copyright (c) 2024 Zenteiq Aitech Innovations Private Limited and
# AiREX Lab, Indian Institute of Science, Bangalore.
# All rights reserved.
#
# This file is part of SciREX
# (Scientific Research and Engineering eXcellence Platform),
# developed jointly by Zenteiq Aitech Innovations and AiREX Lab
# under the guidance of Prof. Sashikumaar Ganesan.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# For any clarifications or special considerations,
# please contact: contact@scirex.org

import json
import os
import sys
import time

# Prevents memory fragmentation
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
# Stops JAX from blindly reserving 90% of your VRAM on startup
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
# os.environ["JAX_ENABLE_X64"] = "True" 
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

os.environ["XLA_FLAGS"] = (
    os.environ.get("XLA_FLAGS", "") + " --xla_gpu_deterministic_ops=true"
)
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import flax
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
import pickle

from configs.darcy_wno2d_config import DarcyWNO2DConfig
from scirex.operators.data.darcy_mat import load_darcy_mat
from scirex.operators.losses import lp_loss
from scirex.operators.models.wno import WNO
from scirex.operators.training import GaussianNormalizer, create_train_state


def make_schedule(config: DarcyWNO2DConfig, steps_per_epoch: int):
    total_steps = max(config.epochs * steps_per_epoch, 1)
    warmup_steps = min(max(total_steps // 20, 1), 500)
    cosine_steps = max(total_steps - warmup_steps, 1)
    return optax.join_schedules(
        schedules=[
            optax.linear_schedule(0.0, config.learning_rate, warmup_steps),
            optax.cosine_decay_schedule(config.learning_rate, cosine_steps, alpha=0.0),
        ],
        boundaries=[warmup_steps],
    )


def plot_loss_curves(history: dict, save_path: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    epochs = range(len(history["train_rel_l2"]))
    ax.semilogy(epochs, history["train_rel_l2"], lw=2, label="Train Rel-L2")
    ax.semilogy(epochs, history["test_rel_l2"], lw=2, ls="--", label="Test Rel-L2")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Relative L2 Error")
    ax.set_title("WNO2D — Darcy Flow Convergence")
    ax.legend()
    ax.grid(True, which="both", ls="-", alpha=0.4)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Loss curve -> {save_path}")


def main() -> None:
    config = DarcyWNO2DConfig()

    rng = jax.random.PRNGKey(config.seed)
    rng, init_rng = jax.random.split(rng)

    x_train, y_train, x_test, y_test = load_darcy_mat(
        train_path=config.train_path,
        test_path=config.test_path,
        n_train=config.n_train,
        n_test=config.n_test,
        subsample_rate=config.subsample_rate,
    )

    n_train_actual = x_train.shape[0]
    nx, ny = x_train.shape[1], x_train.shape[2]
    in_channels = x_train.shape[-1]
    out_channels = y_train.shape[-1]

    x_train_jnp = jnp.asarray(x_train)
    y_train_jnp = jnp.asarray(y_train)
    x_test_jnp = jnp.asarray(x_test)
    y_test_jnp = jnp.asarray(y_test)

    print(
        f"Initializing WNO (hidden_channels={config.hidden_channels}, "
        f"wavelet={config.wavelet}, level={config.level})..."
    )
    print(
        f"Loading Darcy data: train={config.train_path}, test={config.test_path}"
    )
    print(f"Data shapes: x_train={x_train.shape}, y_train={y_train.shape}")

    x_norm = GaussianNormalizer(x_train_jnp) if config.encode_input else None
    y_norm = GaussianNormalizer(y_train_jnp) if config.encode_output else None

    x_tr = x_norm.encode(x_train_jnp) if x_norm else x_train_jnp
    y_tr = y_norm.encode(y_train_jnp) if y_norm else y_train_jnp
    x_te = x_norm.encode(x_test_jnp) if x_norm else x_test_jnp
    test_batch_enc = {"x": x_te, "y": y_norm.encode(y_test_jnp) if y_norm else y_test_jnp}

    model = WNO(
        hidden_channels=config.hidden_channels,
        n_layers=config.n_layers,
        level=config.level,
        size=(nx, ny),
        out_channels=out_channels,
        wavelet=config.wavelet,
        mode=config.mode,
        lifting_channel_ratio=config.lifting_channel_ratio,
        projection_channel_ratio=config.projection_channel_ratio,
        use_grid=False,
        padding=config.padding,
        skip_type=config.skip_type,
    )

    steps_per_epoch = max(n_train_actual // config.batch_size, 1)
    total_steps = config.epochs * steps_per_epoch
    schedule = make_schedule(config, steps_per_epoch)
    tx = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(schedule, weight_decay=config.weight_decay),
    )

    state = create_train_state(
        rng=init_rng,
        model=model,
        input_shape=(config.batch_size, nx, ny, in_channels),
        tx=tx,
    )

    ckpt_dir = os.path.join(PROJECT_ROOT, "experiments", "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, f"{config.run_name}_best.msgpack")

    results_dir = os.path.join(PROJECT_ROOT, "experiments", "results", config.run_name)
    os.makedirs(results_dir, exist_ok=True)
    metrics_path = os.path.join(results_dir, "metrics.json")
    config_path = os.path.join(results_dir, "config.json")
    loss_plot_path = os.path.join(results_dir, "loss_curves.png")

    with open(config_path, "w", encoding="ascii") as fp:
        json.dump(vars(config), fp, indent=2)

    best_test_rel_l2 = float("inf")
    history = {"train_rel_l2": [], "test_rel_l2": []}
    
    # Evolution tracking for animation
    evolution = {
        "ground_truth": np.array(y_test_jnp[0, ..., 0]),
        "predictions": [],
        "steps": []
    }

    @jax.jit
    def train_step(state, batch):
        def loss_fn(params):
            pred = state.apply_fn({"params": params}, batch["x"])
            return lp_loss(pred, batch["y"])

        loss, grads = jax.value_and_grad(loss_fn)(state.params)
        return state.apply_gradients(grads=grads), {"loss": loss}

    print(f"Starting training for {config.epochs} epochs ({total_steps} steps)...")
    rng_key = jax.random.PRNGKey(config.seed + 1)
    total_start_time = time.time()

    for epoch in range(config.epochs):
        t0 = time.time()
        epoch_loss = 0.0

        rng_key, shuffle_key = jax.random.split(rng_key)
        perm = jax.random.permutation(shuffle_key, n_train_actual)
        x_shuf = x_tr[perm]
        y_shuf = y_tr[perm]

        for step in range(steps_per_epoch):
            start = step * config.batch_size
            end = start + config.batch_size
            state, metrics = train_step(
                state, {"x": x_shuf[start:end], "y": y_shuf[start:end]}
            )
            epoch_loss += float(metrics["loss"])

        avg_train = epoch_loss / steps_per_epoch

        pred_enc = state.apply_fn({"params": state.params}, test_batch_enc["x"])
        pred_dec = y_norm.decode(pred_enc) if y_norm else pred_enc
        test_rel_l2 = float(lp_loss(pred_dec, y_test_jnp))

        history["train_rel_l2"].append(avg_train)
        history["test_rel_l2"].append(test_rel_l2)

        if test_rel_l2 < best_test_rel_l2:
            best_test_rel_l2 = test_rel_l2
            with open(ckpt_path, "wb") as fp:
                fp.write(flax.serialization.to_bytes(state.params))

        if epoch % 10 == 0 or epoch == config.epochs - 1:
            current_lr = float(schedule(state.step))
            print(
                f"Epoch {epoch:4d} | Train Rel L2: {avg_train:.6e} | "
                f"Test Rel L2: {test_rel_l2:.6f} | Best Rel L2: {best_test_rel_l2:.6f} | "
                f"LR: {current_lr:.2e} | Time: {time.time() - t0:.2f}s"
            )
            with open(metrics_path, "w", encoding="ascii") as fp:
                json.dump(history, fp, indent=2)
            
            # Save evolution frame
            evolution["predictions"].append(np.array(pred_dec[0, ..., 0]))
            evolution["steps"].append(epoch)

    total_time = time.time() - total_start_time
    print(f"\nTraining Complete. Best Test Rel L2: {best_test_rel_l2:.6f}")
    print(f"Total training time: {total_time:.2f}s ({total_time/60:.2f}m)")

    with open(ckpt_path, "rb") as fp:
        best_params = flax.serialization.from_bytes(state.params, fp.read())

    y_pred_all = []
    n_test_actual = x_test.shape[0]
    for i in range(0, n_test_actual, config.batch_size):
        chunk = x_te[i : i + config.batch_size]
        pred_enc = state.apply_fn({"params": best_params}, chunk)
        pred_dec = y_norm.decode(pred_enc) if y_norm else pred_enc
        y_pred_all.append(np.array(pred_dec))

    y_pred_np = np.concatenate(y_pred_all, axis=0)
    y_test_np = np.array(y_test_jnp)
    diff_all = (y_pred_np - y_test_np).reshape(n_test_actual, -1)
    targ_all = y_test_np.reshape(n_test_actual, -1)
    _ = np.linalg.norm(diff_all, axis=1) / (np.linalg.norm(targ_all, axis=1) + 1e-8)

    plot_loss_curves(history, loss_plot_path)
    
    # Save evolution data
    evo_path = os.path.join(results_dir, f"{config.run_name}_evolution.pkl")
    with open(evo_path, "wb") as f:
        pickle.dump(evolution, f)
    print(f"Evolution data saved to: {evo_path}")
    
    print(f"Loss curves saved to: {results_dir}")


if __name__ == "__main__":
    main()
