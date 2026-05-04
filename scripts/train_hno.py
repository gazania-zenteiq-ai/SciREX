# Copyright (c) 2024 Zenteiq Aitech Innovations Private Limited
# All rights reserved.

import os
import sys

# Force deterministic GPU operations for reproducibility
os.environ["XLA_FLAGS"] = os.environ.get("XLA_FLAGS", "") + " --xla_gpu_deterministic_ops=true"
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"

# Ensure project root is in path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

import jax
import jax.numpy as jnp
import numpy as np
import optax
import flax
from flax import linen as nn
import time
import matplotlib.pyplot as plt
import json
import pandas as pd
from dataclasses import dataclass
from typing import Tuple

from scirex.operators.models.hno import HNO
from scirex.operators.training import create_train_state, UnitGaussianNormalizer
from scirex.operators.losses import lp_loss

# ==========================================
# 1. Configuration
# ==========================================
@dataclass
class HNOConfig:
    # Model architecture
    hidden_channels: int = 32
    n_layers: int = 4
    n_modes: Tuple[int, int] = (16, 16) # (modes_r, modes_theta)
    out_channels: int = 1
    lifting_channel_ratio: int = 2
    projection_channel_ratio: int = 2
    use_channel_mlp: bool = True
    channel_mlp_skip: str = "soft-gating"
    hno_skip: str = "linear"
    use_norm: bool = False
    domain_padding: float = 0.0
    
    # Training
    batch_size: int = 2
    epochs: int = 5
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    seed: int = 42
    
    # Scheduler
    scheduler_type: str = "cosine"
    cosine_decay_epochs: int = 100
    
    # Data resolution - Must perfectly match the grid shape inside your CSV files
    res_r: int = 500
    res_theta: int = 500

# ==========================================
# 2. File Paths for Data
# ==========================================
# MANUALLY INPUT YOUR FILE NAMES HERE:
# Each file represents one sample (e.g. one PDE solution / one grid)
TRAIN_FILES = [
    "data/20250804_stator_magnetOD_28_1_Az_30d_uniform.csv",
    "data/20250804_stator_magnetOD_28_1_Az_60d_uniform.csv",
    "data/20250804_stator_magnetOD_32_1_Az_30d_uniform.csv",
    "data/20250804_stator_magnetOD_36_1_Az_30d_uniform.csv",
    "data/20250804_stator_magnetOD_36_1_Az_60d_uniform.csv",
    "data/20250820_stator_magnetOD_30_1_Az_30d_uniform.csv",
]

TEST_FILES = [
   "data/20250820_stator_magnetOD_34_1_Az_30d_uniform.csv",
   "data/20250804_stator_magnetOD_32_1_Az_60d_uniform.csv",
]

# ==========================================
# 3. Data Loading Function
# ==========================================
def load_hno_data(file_list, config):
    """
    Reads the list of files and constructs the input/output tensors.
    Assuming each file represents a single sample containing columns: r, theta, Az
    """
    inputs = []
    outputs = []
    
    for file_path in file_list:
        print(f"Loading {file_path}...")
        try:
            # Fallback to whitespace delimited if comma separation isn't found
            df = pd.read_csv(file_path)
            if 'Az' not in df.columns:
                df = pd.read_csv(file_path, sep=r'\s+')
            
            # If r and theta are missing, calculate them from x and y
            if 'r' not in df.columns and 'x' in df.columns and 'y' in df.columns:
                df['r'] = np.sqrt(df['x']**2 + df['y']**2)
                df['theta'] = np.arctan2(df['y'], df['x'])
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            continue
            
        # Ensure your rows map correctly to a (res_r x res_theta) 2D grid.
        # If your data is flattened in the CSV, reshape it back into the grid:
        try:
            r_grid = df['r'].values.reshape((config.res_r, config.res_theta))
            theta_grid = df['theta'].values.reshape((config.res_r, config.res_theta))
            Az_grid = df['Az'].values.reshape((config.res_r, config.res_theta))
            if 'file_id' in df.columns:
                file_id_grid = df['file_id'].values.reshape((config.res_r, config.res_theta))
            else:
                file_id_grid = np.ones((config.res_r, config.res_theta))
        except ValueError as e:
            print(f"Reshape error in {file_path}. Ensure it has exactly {config.res_r * config.res_theta} rows.")
            raise e
        
        # Define the Input and Output for the Neural Operator.
        # Here we use (theta, r, file_id) as the input features (Channels = 3)
        # And we predict Az as the output (Channels = 1)
        input_field = np.stack([theta_grid, r_grid, file_id_grid], axis=-1)
        output_field = np.expand_dims(Az_grid, axis=-1)
        
        inputs.append(input_field)
        outputs.append(output_field)
        
    if len(inputs) == 0:
        print("WARNING: No files provided in the list. Generating dummy data for dry-run testing.")
        dummy_in = np.random.randn(2, config.res_r, config.res_theta, 3)
        dummy_out = np.random.randn(2, config.res_r, config.res_theta, 1)
        return dummy_in, dummy_out

    return np.stack(inputs), np.stack(outputs)

# ==========================================
# 4. Optimizer Schedule
# ==========================================
def make_schedule(config: HNOConfig, steps_per_epoch: int):
    total_steps = config.epochs * steps_per_epoch
    warmup_steps = min(310, max(1, total_steps // 10))
    
    cosine_decay_steps = config.cosine_decay_epochs * steps_per_epoch - warmup_steps
    cosine_decay_steps = max(cosine_decay_steps, 1)
    
    cosine_schedule = optax.cosine_decay_schedule(
        init_value=config.learning_rate,
        decay_steps=cosine_decay_steps,
        alpha=0.0
    )
    schedule = optax.join_schedules(
        schedules=[
            optax.linear_schedule(0.0, config.learning_rate, warmup_steps),
            cosine_schedule
        ],
        boundaries=[warmup_steps]
    )
    return schedule

# ==========================================
# 5. Main Training Loop
# ==========================================
def main():
    config = HNOConfig()
    
    # Load Data
    print("--- Loading Training Data ---")
    x_train, y_train = load_hno_data(TRAIN_FILES, config)
    print("--- Loading Testing Data ---")
    x_test, y_test = load_hno_data(TEST_FILES, config)
    
    n_train = x_train.shape[0]
    n_test = x_test.shape[0]
    
    print(f"Data shapes: x_train={x_train.shape}, y_train={y_train.shape}")
    
    # Normalize Data using SciREX tools
    x_normalizer = UnitGaussianNormalizer(x_train)
    y_normalizer = UnitGaussianNormalizer(y_train)
    
    x_train_encoded = jnp.asarray(x_normalizer.encode(x_train))
    y_train_encoded = jnp.asarray(y_normalizer.encode(y_train))
    x_test_encoded = jnp.asarray(x_normalizer.encode(x_test))
    y_test_encoded = jnp.asarray(y_normalizer.encode(y_test))
    
    test_batch_encoded = {"x": x_test_encoded, "y": y_test_encoded}
    
    # Setup PRNG and Model
    rng = jax.random.PRNGKey(config.seed)
    rng, init_rng = jax.random.split(rng)
    
    print(f"Initializing HNO (hidden_channels={config.hidden_channels}, modes={config.n_modes})...")
    model = HNO(
        hidden_channels=config.hidden_channels, 
        n_layers=config.n_layers, 
        n_modes=config.n_modes, 
        out_channels=config.out_channels,
        lifting_channel_ratio=config.lifting_channel_ratio,
        projection_channel_ratio=config.projection_channel_ratio,
        use_grid=False, # Grid coords are already passed in via load_hno_data
        use_norm=config.use_norm,
        hno_skip=config.hno_skip,
        channel_mlp_skip=config.channel_mlp_skip,
        use_channel_mlp=config.use_channel_mlp,
        padding=config.domain_padding,
        activation=nn.gelu,
        order=0, # Bessel order 0
        r_axis=1,
        theta_axis=2
    )
    
    steps_per_epoch = max(1, n_train // config.batch_size)
    schedule = make_schedule(config, steps_per_epoch)
    
    input_shape = (config.batch_size, config.res_r, config.res_theta, x_train.shape[-1])
    
    state = create_train_state(
        rng=init_rng, 
        model=model, 
        input_shape=input_shape, 
        learning_rate=schedule, 
        weight_decay=config.weight_decay
    )
    
    # Paths for saving checkpoints and plots
    ckpt_dir = os.path.join(project_root, "experiments/checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, "hno_params.pkl")

    results_dir = os.path.join(project_root, "experiments/results/hno_training")
    os.makedirs(results_dir, exist_ok=True)

    best_rel_l2 = float("inf")
    history = {"train_rel_l2": [], "test_rel_l2": []}

    # JIT-compiled training step
    @jax.jit
    def train_step(state, batch):
        def loss_fn(params):
            pred_encoded = state.apply_fn({"params": params}, batch["x"])
            # Pure Data-Driven Lp loss for training (can add PDE loss if governing equation is known)
            return lp_loss(pred_encoded, batch["y"])

        grad_fn = jax.value_and_grad(loss_fn)
        loss, grads = grad_fn(state.params)
        state = state.apply_gradients(grads=grads)
        return state, {"loss": loss}

    print(f"Starting training for {config.epochs} epochs...")
    rng_key = jax.random.PRNGKey(config.seed + 1)
    
    _total_start_time = time.time()
    for epoch in range(config.epochs):
        epoch_start_time = time.time()
        epoch_loss = 0.0
        
        # Shuffle batches
        rng_key, shuffle_key = jax.random.split(rng_key)
        perm = jax.random.permutation(shuffle_key, n_train)
        x_shuffled = x_train_encoded[perm]
        y_shuffled = y_train_encoded[perm]
        
        for step in range(steps_per_epoch):
            start_idx = step * config.batch_size
            end_idx = min(start_idx + config.batch_size, n_train)
            
            # Skip incomplete batches to keep JIT shapes static
            if end_idx - start_idx < config.batch_size and n_train >= config.batch_size:
                continue
                
            batch = {"x": x_shuffled[start_idx:end_idx], "y": y_shuffled[start_idx:end_idx]}
            state, metrics = train_step(state, batch)
            epoch_loss += float(metrics["loss"])
            
        epoch_time = time.time() - epoch_start_time
        avg_train_loss = epoch_loss / steps_per_epoch
        
        # Evaluate on Test Set
        test_pred_encoded = state.apply_fn({"params": state.params}, test_batch_encoded["x"])
        test_pred_decoded = y_normalizer.decode(test_pred_encoded)
        v_test_l2 = float(lp_loss(test_pred_decoded, y_test))

        history["train_rel_l2"].append(avg_train_loss)
        history["test_rel_l2"].append(v_test_l2)
        
        # Save best model
        if v_test_l2 < best_rel_l2:
            best_rel_l2 = v_test_l2
            with open(ckpt_path, "wb") as f:
                f.write(flax.serialization.to_bytes(state.params))
        
        current_lr = schedule(state.step)
        
        # Print logs
        if epoch % 5 == 0 or epoch == config.epochs - 1:
            print(f"Epoch {epoch:4d} | Train Rel L2: {avg_train_loss:.6e} | "
                  f"Test Rel L2: {v_test_l2:.6f} | Best Rel L2: {best_rel_l2:.6f} | "
                  f"LR: {float(current_lr):.2e} | Time: {epoch_time:.2f}s")
            
            # Save metrics dynamically
            with open(os.path.join(results_dir, "hno_metrics.json"), "w") as f:
                json.dump(history, f, indent=4)

    total_time = time.time() - _total_start_time
    print(f"\nTraining Complete. Best Test Rel L2: {best_rel_l2:.6f}")
    print(f"Total time: {total_time:.2f}s")

    # Plot Convergence
    plt.figure(figsize=(8, 8))
    plt.semilogy(range(len(history["train_rel_l2"])), history["train_rel_l2"], label='Train Rel L2')
    plt.semilogy(range(len(history["test_rel_l2"])), history["test_rel_l2"], label='Test Rel L2', linestyle='--')
    plt.xlabel('Epoch')
    plt.ylabel('Relative L2 Error')
    plt.title('Hankel Neural Operator (HNO): Convergence')
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.legend()
    plt.savefig(os.path.join(results_dir, "hno_losses.png"), dpi=150)
    print(f"Loss curves saved to: {results_dir}")

if __name__ == "__main__":
    main()
