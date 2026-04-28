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
import jax
import jax.numpy as jnp
import optax
import time
import numpy as np
from flax.training import train_state
from absl import app
from absl import flags
from ml_collections import config_flags

# Add the root directory to path with highest priority to override pip installed src/scirex
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scirex.operators.models.transolver import Transolver

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "configs/car_design_transolver_config.py", "Training configuration.", lock_config=True)

class CarDataset:
    """Simple dataloader for preprocessed Car Design ShapeNet numpy arrays with Normalization."""
    def __init__(self, data_dir, compute_norm=True):
        self.data_dir = data_dir
        self.samples = []
        if os.path.exists(data_dir):
            for param_dir in os.listdir(data_dir):
                param_path = os.path.join(data_dir, param_dir)
                if os.path.isdir(param_path):
                    for sample in os.listdir(param_path):
                        self.samples.append(os.path.join(param_path, sample))
                        
        self.mean_in, self.mean_out = 0.0, 0.0
        self.std_in, self.std_out = 1.0, 1.0
        
        if compute_norm and len(self.samples) > 0:
            self._compute_statistics()
            
    def _compute_statistics(self):
        print("Computing dataset statistics for normalization...")
        old_length = 0
        mean_in, mean_out = 0.0, 0.0
        std_in, std_out = 0.0, 0.0
        
        # Pass 1: Mean
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
                
        # Pass 2: Std
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
                std_in += (((x - mean_in) ** 2).sum(axis=0) - x.shape[0] * std_in) / new_length
                std_out += (((y - mean_out) ** 2).sum(axis=0) - y.shape[0] * std_out) / new_length
                old_length = new_length
                
        self.mean_in = mean_in
        self.mean_out = mean_out
        self.std_in = np.sqrt(std_in)
        self.std_out = np.sqrt(std_out)
        print("Statistics computed.")

    def __len__(self):
        return len(self.samples)

    def generator(self, batch_size=1, shuffle=True):
        indices = np.arange(len(self.samples))
        if shuffle:
            np.random.shuffle(indices)
        
        for i in range(0, len(self.samples), batch_size):
            batch_indices = indices[i:i + batch_size]
            batch_x, batch_y, batch_surf = [], [], []
            
            for idx in batch_indices:
                path = self.samples[idx]
                x = np.load(os.path.join(path, "x.npy"))
                y = np.load(os.path.join(path, "y.npy"))
                surf = np.load(os.path.join(path, "surf.npy"))
                
                # Normalize
                x = (x - self.mean_in) / (self.std_in + 1e-8)
                y = (y - self.mean_out) / (self.std_out + 1e-8)
                
                batch_x.append(x)
                batch_y.append(y)
                batch_surf.append(surf)
                
            yield jnp.array(batch_x), jnp.array(batch_y), jnp.array(batch_surf)


def create_train_state(rng, config, model, dummy_x, steps_per_epoch):
    """Initializes the model and the training state."""
    params = model.init(rng, dummy_x, deterministic=False)['params']
    
    # We use Optax for the optimizer with cosine onecycle scheduling similar to original repo
    total_steps = steps_per_epoch * config.training.num_epochs
    lr_schedule = optax.cosine_onecycle_schedule(
        transition_steps=total_steps,
        peak_value=config.training.learning_rate,
        pct_start=0.3, # Usually 30% warmup
        final_div_factor=1000.0,
    )
    
    tx = optax.adamw(
        learning_rate=lr_schedule,
        weight_decay=config.training.weight_decay
    )
    
    return train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx
    )


@jax.jit
def train_step(state, batch_x, batch_y, batch_surf, reg_weight):
    """Computes loss and updates weights."""
    def loss_fn(params):
        preds = state.apply_fn({'params': params}, batch_x, deterministic=False)
        
        # Preds and y shape: (B, N, 4) -> (v_x, v_y, v_z, P)
        # We process point-wise mean loss
        mse_loss = jnp.square(preds - batch_y)
        
        # Velocity Loss (all points, first 3 channels)
        loss_velo = jnp.mean(mse_loss[..., :-1])
        
        # Pressure Loss (only on surface points, last channel)
        # Add a small epsilon to avoid NaN if no surface points exist
        surf_mask = batch_surf[..., None]
        press_loss_masked = mse_loss[..., -1:] * surf_mask
        loss_press = jnp.sum(press_loss_masked) / (jnp.sum(surf_mask) + 1e-8)
        
        total_loss = loss_velo + reg_weight * loss_press
        return total_loss, {"loss_velo": loss_velo, "loss_press": loss_press, "loss": total_loss}

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, metrics), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    
    return state, metrics


def main(argv):
    config = FLAGS.config
    
    # Initialize Dataloader
    dataset = CarDataset(config.data.data_dir, compute_norm=True)
    print(f"Loaded {len(dataset)} samples from {config.data.data_dir}")
    
    if len(dataset) == 0:
        print("No preprocessed data found. Please run the CarDataProcessor first.")
        return

    steps_per_epoch = max(1, len(dataset) // config.data.batch_size)

    # Initialize model
    model = Transolver(
        hidden_channels=config.model.hidden_channels,
        out_channels=config.model.out_channels,
        n_layers=config.model.n_layers,
        n_heads=config.model.n_heads,
        dim_head=config.model.dim_head,
        slice_num=config.model.slice_num,
        mlp_ratio=config.model.mlp_ratio,
        dropout=config.model.dropout
    )
    
    rng = jax.random.PRNGKey(0)
    
    # Get a dummy input to initialize the state
    gen = dataset.generator(batch_size=config.data.batch_size)
    dummy_x, _, _ = next(gen)
    
    state = create_train_state(rng, config, model, dummy_x, steps_per_epoch)

    print("Starting Training Loop...")
    for epoch in range(1, config.training.num_epochs + 1):
        start_time = time.time()
        
        epoch_loss = 0.0
        epoch_velo_loss = 0.0
        epoch_press_loss = 0.0
        num_batches = 0
        
        for batch_x, batch_y, batch_surf in dataset.generator(batch_size=config.data.batch_size, shuffle=config.data.shuffle):
            state, metrics = train_step(state, batch_x, batch_y, batch_surf, config.training.reg_weight)
            
            epoch_loss += metrics["loss"].item()
            epoch_velo_loss += metrics["loss_velo"].item()
            epoch_press_loss += metrics["loss_press"].item()
            num_batches += 1
            
        epoch_loss /= num_batches
        epoch_velo_loss /= num_batches
        epoch_press_loss /= num_batches
        
        end_time = time.time()
        
        print(f"Epoch {epoch:03d} | Time: {end_time - start_time:.2f}s | "
              f"Loss: {epoch_loss:.6f} (Velo: {epoch_velo_loss:.6f}, Press: {epoch_press_loss:.6f})")

if __name__ == "__main__":
    app.run(main)
