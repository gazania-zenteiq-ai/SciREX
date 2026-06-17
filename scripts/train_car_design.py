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

import os
import sys
import json
import jax
import jax.numpy as jnp
import optax
import time
import numpy as np
from flax.training import train_state
from absl import app
from absl import flags
from ml_collections import config_flags

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scirex.operators.models.transolver import Transolver

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file(
    "config",
    os.path.join(os.path.dirname(__file__), "..", "configs", "car_design_transolver_config.py"),
    "Training configuration.",
    lock_config=True,
)


class CarDataset:
    """
    Dataloader for preprocessed Car Design ShapeNet numpy arrays.

    Normalization is computed from training data only. For the validation
    split, pass coef_norm=(mean_in, std_in, mean_out, std_out) from the
    training dataset so both splits share the same statistics — matching
    the reference Transolver implementation.
    """

    def __init__(self, data_dir, param_dirs, coef_norm=None):
        """
        Args:
            data_dir:   Root directory containing param0..param8 folders.
            param_dirs: List of param folder names to include (e.g. ['param0']).
            coef_norm:  (mean_in, std_in, mean_out, std_out) from the training
                        split. If None, statistics are computed from this split.
        """
        self.data_dir = data_dir
        self.samples = []

        for param in sorted(param_dirs):
            param_path = os.path.join(data_dir, param)
            if not os.path.isdir(param_path):
                continue
            for sample in sorted(os.listdir(param_path)):
                sample_path = os.path.join(param_path, sample)
                if os.path.isdir(sample_path) and os.path.exists(
                    os.path.join(sample_path, "x.npy")
                ):
                    self.samples.append(sample_path)

        if coef_norm is not None:
            self.mean_in, self.std_in, self.mean_out, self.std_out = coef_norm
        else:
            self.mean_in = self.mean_out = 0.0
            self.std_in = self.std_out = 1.0
            if len(self.samples) > 0:
                self._compute_statistics()

    def get_coef_norm(self):
        return (self.mean_in, self.std_in, self.mean_out, self.std_out)

    def _compute_statistics(self):
        print("Computing normalization statistics on training split...")
        old_length = 0
        mean_in, mean_out = 0.0, 0.0
        std_in, std_out = 0.0, 0.0

        # Pass 1: running mean
        for i, path in enumerate(self.samples):
            x = np.load(os.path.join(path, "x.npy"))
            y = np.load(os.path.join(path, "y.npy"))
            if i == 0:
                old_length = x.shape[0]
                mean_in = x.mean(axis=0)
                mean_out = y.mean(axis=0)
            else:
                new_length = old_length + x.shape[0]
                mean_in += (x.sum(axis=0) - x.shape[0] * mean_in) / new_length
                mean_out += (y.sum(axis=0) - y.shape[0] * mean_out) / new_length
                old_length = new_length

        # Pass 2: running variance
        old_length = 0
        for i, path in enumerate(self.samples):
            x = np.load(os.path.join(path, "x.npy"))
            y = np.load(os.path.join(path, "y.npy"))
            if i == 0:
                old_length = x.shape[0]
                std_in = ((x - mean_in) ** 2).sum(axis=0) / old_length
                std_out = ((y - mean_out) ** 2).sum(axis=0) / old_length
            else:
                new_length = old_length + x.shape[0]
                std_in += (
                    ((x - mean_in) ** 2).sum(axis=0) - x.shape[0] * std_in
                ) / new_length
                std_out += (
                    ((y - mean_out) ** 2).sum(axis=0) - y.shape[0] * std_out
                ) / new_length
                old_length = new_length

        self.mean_in = mean_in
        self.mean_out = mean_out
        self.std_in = np.sqrt(std_in)
        self.std_out = np.sqrt(std_out)
        print("Statistics computed.")

    def __len__(self):
        return len(self.samples)

    def get_sample(self, idx):
        """Return a single normalized sample as jnp arrays with a batch dim."""
        path = self.samples[idx]
        x = np.load(os.path.join(path, "x.npy"))
        y = np.load(os.path.join(path, "y.npy"))
        surf = np.load(os.path.join(path, "surf.npy"))
        x = (x - self.mean_in) / (self.std_in + 1e-8)
        y = (y - self.mean_out) / (self.std_out + 1e-8)
        return jnp.array(x[None]), jnp.array(y[None]), jnp.array(surf[None])

    def generator(self, batch_size=1, shuffle=True):
        indices = np.arange(len(self.samples))
        if shuffle:
            np.random.shuffle(indices)
        for i in range(0, len(self.samples), batch_size):
            batch_x, batch_y, batch_surf = [], [], []
            for idx in indices[i : i + batch_size]:
                path = self.samples[idx]
                x = np.load(os.path.join(path, "x.npy"))
                y = np.load(os.path.join(path, "y.npy"))
                surf = np.load(os.path.join(path, "surf.npy"))
                x = (x - self.mean_in) / (self.std_in + 1e-8)
                y = (y - self.mean_out) / (self.std_out + 1e-8)
                batch_x.append(x)
                batch_y.append(y)
                batch_surf.append(surf)
            yield jnp.array(batch_x), jnp.array(batch_y), jnp.array(batch_surf)


def create_train_state(rng, config, model, dummy_x, steps_per_epoch):
    params = model.init(rng, dummy_x, deterministic=False)["params"]
    total_steps = steps_per_epoch * config.training.num_epochs
    lr_schedule = optax.cosine_onecycle_schedule(
        transition_steps=total_steps,
        peak_value=config.training.learning_rate,
        pct_start=0.3,
        final_div_factor=1000.0,
    )
    tx = optax.adamw(learning_rate=lr_schedule, weight_decay=config.training.weight_decay)
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)


@jax.jit
def train_step(state, batch_x, batch_y, batch_surf, reg_weight):
    def loss_fn(params):
        preds = state.apply_fn({"params": params}, batch_x, deterministic=False)
        mse = jnp.square(preds - batch_y)
        loss_velo = jnp.mean(mse[..., :-1])
        surf_mask = batch_surf[..., None]
        loss_press = jnp.sum(mse[..., -1:] * surf_mask) / (jnp.sum(surf_mask) + 1e-8)
        total = loss_velo + reg_weight * loss_press
        return total, {"loss": total, "loss_velo": loss_velo, "loss_press": loss_press}

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (_, metrics), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, metrics


@jax.jit
def eval_step(state, batch_x):
    """Forward pass with dropout disabled, no gradient tracking."""
    return state.apply_fn({"params": state.params}, batch_x, deterministic=True)


def evaluate(state, val_dataset):
    """
    Compute validation metrics in de-normalized (original) space, matching
    the reference Transolver main_evaluation.py:

      - Relative L2 pressure  : surface points only  (last channel)
      - Relative L2 velocity  : exterior points only  (first 3 channels)
      - RMSE pressure         : surface points only
      - RMSE velocity         : exterior points only, per component then scalar
    """
    mean_out = np.array(val_dataset.mean_out)  # (4,)
    std_out = np.array(val_dataset.std_out)    # (4,)

    rel_l2_press_list = []
    rel_l2_velo_list = []
    rmse_press_list = []
    rmse_velo_list = []

    for i in range(len(val_dataset)):
        x, y_norm, surf = val_dataset.get_sample(i)
        pred_norm = eval_step(state, x)  # (1, N, 4)

        # De-normalize to original scale
        pred = np.array(pred_norm[0]) * std_out + mean_out  # (N, 4)
        gt = np.array(y_norm[0]) * std_out + mean_out       # (N, 4)
        surf_mask = np.array(surf[0]).astype(bool)           # (N,)
        ext_mask = ~surf_mask

        # Pressure — surface only
        pred_press = pred[surf_mask, -1]
        gt_press = gt[surf_mask, -1]
        rel_l2_press = np.linalg.norm(pred_press - gt_press) / (
            np.linalg.norm(gt_press) + 1e-8
        )
        rmse_press = np.sqrt(np.mean(np.square(pred_press - gt_press)))

        # Velocity — exterior only
        pred_velo = pred[ext_mask, :-1]
        gt_velo = gt[ext_mask, :-1]
        rel_l2_velo = np.linalg.norm(pred_velo - gt_velo) / (
            np.linalg.norm(gt_velo) + 1e-8
        )
        # Per-component RMSE, then scalar: sqrt(mean(rmse_per_component^2))
        rmse_velo_comp = np.sqrt(np.mean(np.square(pred_velo - gt_velo), axis=0))

        rel_l2_press_list.append(rel_l2_press)
        rel_l2_velo_list.append(rel_l2_velo)
        rmse_press_list.append(rmse_press)
        rmse_velo_list.append(rmse_velo_comp)

    rmse_velo_per_comp = np.sqrt(np.mean(np.square(np.stack(rmse_velo_list)), axis=0))
    return {
        "rel_l2_press": float(np.mean(rel_l2_press_list)),
        "rel_l2_velo": float(np.mean(rel_l2_velo_list)),
        "rmse_press": float(np.mean(rmse_press_list)),
        "rmse_velo_x": float(rmse_velo_per_comp[0]),
        "rmse_velo_y": float(rmse_velo_per_comp[1]),
        "rmse_velo_z": float(rmse_velo_per_comp[2]),
        "rmse_velo_scalar": float(np.sqrt(np.mean(np.square(rmse_velo_per_comp)))),
    }


def save_metrics(metrics_log, out_path):
    with open(out_path, "w") as f:
        json.dump(metrics_log, f, indent=2)


def main(argv):
    config = FLAGS.config
    data_dir = config.data.data_dir

    # --- Train / Val split (fold-based, matching reference) ---
    all_params = sorted(
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d)) and d.startswith("param")
    )
    if not all_params:
        print(f"No param* directories found in {data_dir}")
        return

    fold_id = config.data.fold_id
    val_params = [all_params[fold_id]]
    train_params = [p for p in all_params if p != all_params[fold_id]]
    print(f"Train params: {train_params}")
    print(f"Val   params: {val_params}  (fold_id={fold_id})")

    # Training split — computes normalization stats
    train_ds = CarDataset(data_dir, train_params)
    print(f"Train samples: {len(train_ds)}")
    if len(train_ds) == 0:
        print("No training data found. Run preprocess_car_design.py first.")
        return

    # Validation split — reuses training normalization stats
    val_ds = CarDataset(data_dir, val_params, coef_norm=train_ds.get_coef_norm())
    print(f"Val   samples: {len(val_ds)}")

    steps_per_epoch = max(1, len(train_ds) // config.data.batch_size)

    # --- Model ---
    model = Transolver(
        hidden_channels=config.model.hidden_channels,
        out_channels=config.model.out_channels,
        n_layers=config.model.n_layers,
        n_heads=config.model.n_heads,
        dim_head=config.model.dim_head,
        slice_num=config.model.slice_num,
        mlp_ratio=config.model.mlp_ratio,
        dropout=config.model.dropout,
    )

    rng = jax.random.PRNGKey(0)
    dummy_x, _, _ = next(train_ds.generator(batch_size=config.data.batch_size))
    state = create_train_state(rng, config, model, dummy_x, steps_per_epoch)

    # --- Metrics log (written after every epoch for safe incremental saving) ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    metrics_path = os.path.join(script_dir, "..", "metrics_scirex.json")
    metrics_log = {
        "framework": "SciREX (JAX/Flax)",
        "fold_id": fold_id,
        "num_epochs": config.training.num_epochs,
        "val_iter": config.training.val_iter,
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
        "epochs": [],
    }

    # --- Training loop ---
    print("Starting training...")
    for epoch in range(1, config.training.num_epochs + 1):
        t0 = time.time()
        epoch_loss = epoch_velo = epoch_press = 0.0
        num_batches = 0

        for bx, by, bs in train_ds.generator(
            batch_size=config.data.batch_size, shuffle=config.data.shuffle
        ):
            state, metrics = train_step(
                state, bx, by, bs, config.training.reg_weight
            )
            epoch_loss += metrics["loss"].item()
            epoch_velo += metrics["loss_velo"].item()
            epoch_press += metrics["loss_press"].item()
            num_batches += 1

        epoch_loss /= num_batches
        epoch_velo /= num_batches
        epoch_press /= num_batches
        elapsed = time.time() - t0

        epoch_record = {
            "epoch": epoch,
            "train_loss": epoch_loss,
            "train_velo": epoch_velo,
            "train_press": epoch_press,
            "time_s": round(elapsed, 2),
        }

        print(
            f"Epoch {epoch:03d} | {elapsed:.1f}s | "
            f"Loss: {epoch_loss:.6f}  (Velo: {epoch_velo:.6f}, Press: {epoch_press:.6f})",
            end="",
        )

        # --- Validation ---
        if len(val_ds) > 0 and epoch % config.training.val_iter == 0:
            val_t0 = time.time()
            val_metrics = evaluate(state, val_ds)
            val_elapsed = time.time() - val_t0

            epoch_record["val_rel_l2_press"] = val_metrics["rel_l2_press"]
            epoch_record["val_rel_l2_velo"] = val_metrics["rel_l2_velo"]
            epoch_record["val_rmse_press"] = val_metrics["rmse_press"]
            epoch_record["val_rmse_velo_x"] = val_metrics["rmse_velo_x"]
            epoch_record["val_rmse_velo_y"] = val_metrics["rmse_velo_y"]
            epoch_record["val_rmse_velo_z"] = val_metrics["rmse_velo_z"]
            epoch_record["val_rmse_velo_scalar"] = val_metrics["rmse_velo_scalar"]
            epoch_record["val_time_s"] = round(val_elapsed, 2)

            print(
                f"\n          Val ({val_elapsed:.1f}s) | "
                f"Rel-L2 press: {val_metrics['rel_l2_press']:.4f}  "
                f"velo: {val_metrics['rel_l2_velo']:.4f} | "
                f"RMSE press: {val_metrics['rmse_press']:.4e}  "
                f"velo: [{val_metrics['rmse_velo_x']:.4e}, "
                f"{val_metrics['rmse_velo_y']:.4e}, "
                f"{val_metrics['rmse_velo_z']:.4e}]  "
                f"scalar: {val_metrics['rmse_velo_scalar']:.4e}",
                end="",
            )

        print()

        metrics_log["epochs"].append(epoch_record)
        save_metrics(metrics_log, metrics_path)

    print(f"\nMetrics saved to {metrics_path}")


if __name__ == "__main__":
    app.run(main)
