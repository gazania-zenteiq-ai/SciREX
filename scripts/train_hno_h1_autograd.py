# Copyright (c) 2024 Zenteiq Aitech Innovations Private Limited and
# AiREX Lab, Indian Institute of Science, Bangalore.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0

"""
train_hno_h1_autograd.py
========================
Train the Hankel Neural Operator (HNO) to predict Az on uniform polar grids,
using the autograd-based H1 Sobolev loss from sharad_stuff/h1_loss_hno_autograd.py.

Data source (30d only):
    Project_data/grid_hno-20260507T104948Z-3-001/grid_hno/
    Files filtered by suffix: *_Az_30d_uniform.csv  (paired with *_MagB_30d_uniform.csv)

Grid layout (from CSV inspection):
    - rows: 16512 = 129 r-points × 128 theta-points
    - Input channels (index order): [theta, r, magnet_od]
      → theta at ch-idx 0, r at ch-idx 1

H1 loss wiring:
    - The loss function needs:
        * theta_grid  : (batch, N_r, N_theta, 1)  – the θ values
        * B_r_true    : (batch, N_r, N_theta, 1)  – ∂Az/∂r    (approximated from MagB)
        * B_theta_true: (batch, N_r, N_theta, 1)  – (1/r)∂Az/∂θ (approximated from MagB)
      MagB gives |B| = sqrt(B_r² + B_z²).  We can't separate B_r from B_theta
      from a single scalar |B| without additional data, so we pass the isotropic
      decomposition:
          B_r_true     ≈ |B| / sqrt(2)
          B_theta_true ≈ |B| / sqrt(2)
      The loss then minimises:
          RelL2(B_r_pred² + sin²θ·B_θ_pred²,  0.5·|B|² + 0.5·sin²θ·|B|²)
                                               = RelL2(E_pred,  0.5(1+sin²θ)|B|²)
      This is an approximation; if exact component data becomes available,
      replace the decomposition below with real values.

Train/Test split:
    - Test:  magnetOD_32
    - Train: everything else (28, 30, 34, 36) – 30d files only
"""

import os
import sys
import re
import time
import pickle
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
import optax
import flax
from flax import linen as nn
from dataclasses import dataclass, field
from typing import Tuple, List, Optional

# ---------------------------------------------------------------------------
# Project root
# ---------------------------------------------------------------------------
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from scirex.operators.models.hno import HNO
from scirex.operators.training import create_train_state, UnitGaussianNormalizer
from scirex.operators.losses import lp_loss
from sharad_stuff.h1_loss_hno_autograd import h1_loss_hno_autograd_subsampled

# ============================================================
# 1.  Configuration
# ============================================================
@dataclass
class HNOConfig:
    # ---- model ----
    hidden_channels: int = 128
    n_layers: int = 4
    n_modes: Tuple[int, int] = (16, 16)   # (modes_r, modes_theta)
    out_channels: int = 1
    lifting_channel_ratio: int = 2
    projection_channel_ratio: int = 2
    use_channel_mlp: bool = True
    channel_mlp_skip: str = "linear"
    hno_skip: str = "linear"
    use_norm: bool = False
    domain_padding: float = 0.024

    # ---- training ----
    batch_size: int = 1
    epochs: int = 500
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    seed: int = 42

    # ---- LR schedule ----
    cosine_decay_epochs: int = 500
    warmup_fraction: float = 0.06      # fraction of total steps used for warmup

    # ---- H1 loss ----
    data_weight: float = 1.0
    deriv_weight: float = 0.1          # weight on derivative term
    subsample_n: int = 128             # spatial points sampled per batch element

    # ---- data ----
    res_r: int = 129
    res_theta: int = 128
    # input channel indices (must match input_field construction below)
    r_ch_idx: int = 1
    theta_ch_idx: int = 0

    # ---- paths ----
    data_dir: str = os.path.join(
        project_root,
        "Project_data/grid_hno-20260507T104948Z-3-001/grid_hno"
    )
    results_dir: str = os.path.join(
        project_root, "experiments/results/hno_h1_autograd"
    )
    ckpt_dir: str = os.path.join(
        project_root, "experiments/checkpoints/hno_h1_autograd"
    )


# ============================================================
# 2.  File Discovery  (30d only)
# ============================================================

def discover_files(config: HNOConfig) -> Tuple[List[str], List[str]]:
    """Return (train_az_files, test_az_files) – only *_30d_uniform.csv Az files."""
    data_dir = config.data_dir
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    az_files = sorted(
        os.path.join(data_dir, f)
        for f in os.listdir(data_dir)
        if f.endswith("_Az_30d_uniform.csv")
    )

    if not az_files:
        raise RuntimeError(f"No *_Az_30d_uniform.csv files found in {data_dir}")

    train = [f for f in az_files if "magnetOD_32" not in f]
    test  = [f for f in az_files if "magnetOD_32" in f]
    return train, test


# ============================================================
# 3.  Data Loading
# ============================================================

def _magb_col(df: pd.DataFrame, az_path: str) -> str:
    """Return name of the |B| magnitude column, with fallbacks."""
    if "B" in df.columns:
        return "B"
    if "Ax" in df.columns:
        print(f"  [warn] 'B' not found in MagB file for {os.path.basename(az_path)}; "
              f"using 'Ax' as fallback.")
        return "Ax"
    raise KeyError(
        f"No suitable B-magnitude column in MagB file for {az_path}. "
        f"Columns: {df.columns.tolist()}"
    )


def load_dataset(
    az_files: List[str],
    config: HNOConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns
    -------
    x : (N, res_r, res_theta, 3)  – [theta, r, magnet_od]
    y : (N, res_r, res_theta, 1)  – Az
    theta_grid : (N, res_r, res_theta, 1)
    B_r_true   : (N, res_r, res_theta, 1)  – isotropic decomp of |B|
    B_t_true   : (N, res_r, res_theta, 1)  – isotropic decomp of |B|
    """
    xs, ys, thetas, brs, bts = [], [], [], [], []
    NR, NT = config.res_r, config.res_theta

    for az_path in az_files:
        magb_path = az_path.replace("_Az_", "_MagB_")
        if not os.path.isfile(magb_path):
            print(f"[skip] MagB file not found for: {os.path.basename(az_path)}")
            continue

        print(f"  Loading {os.path.basename(az_path)} + {os.path.basename(magb_path)}")

        try:
            df_az   = pd.read_csv(az_path)
            df_magb = pd.read_csv(magb_path)
        except Exception as e:
            print(f"  [error] Reading CSVs: {e}")
            continue

        actual_rows = len(df_az)
        if actual_rows != NR * NT:
            print(f"  [warn] Expected {NR*NT} rows, got {actual_rows} – skipping.")
            continue

        try:
            r_grid     = df_az["r"].values.reshape(NT, NR).T
            theta_grid = df_az["theta"].values.reshape(NT, NR).T
            Az_grid    = df_az["Az"].values.reshape(NT, NR).T
            
            b_col     = _magb_col(df_magb, az_path)
            B_mag_grid = df_magb[b_col].values.reshape(NT, NR).T
        except Exception as e:
            print(f"  [error] Processing {os.path.basename(az_path)}: {e}")
            import traceback; traceback.print_exc()
            continue

        # magnet_od as scalar condition embedded on the grid
        m = re.search(r"magnetOD_(\d+)", az_path)
        magnet_od = float(m.group(1)) if m else 1.0
        param_grid = np.full((NR, NT), magnet_od, dtype=np.float32)

        # -----------------------------------------------------------
        # Isotropic decomposition of |B| into (B_r, B_theta) proxies.
        # |B|² = B_r² + (B_theta/r)²  ← cylindrical
        # We assume equal energy split: B_r ≈ B_theta/r ≈ |B|/sqrt(2)
        # -----------------------------------------------------------
        B_r_proxy = B_mag_grid / np.sqrt(2.0)      # (NR, NT)
        B_t_proxy = B_mag_grid / np.sqrt(2.0)      # same, proxied

        # Input: [theta, r, magnet_od]  (channel layout match r_ch_idx / theta_ch_idx)
        x_i = np.stack([theta_grid, r_grid, param_grid], axis=-1).astype(np.float32)
        y_i = Az_grid[..., None].astype(np.float32)
        t_i = theta_grid[..., None].astype(np.float32)
        br_i = B_r_proxy[..., None].astype(np.float32)
        bt_i = B_t_proxy[..., None].astype(np.float32)

        xs.append(x_i); ys.append(y_i); thetas.append(t_i)
        brs.append(br_i); bts.append(bt_i)

    if not xs:
        raise RuntimeError("No samples loaded. Check data paths and file format.")

    return (
        np.stack(xs),
        np.stack(ys),
        np.stack(thetas),
        np.stack(brs),
        np.stack(bts),
    )


# ============================================================
# 4.  LR Schedule
# ============================================================

def make_schedule(config: HNOConfig, steps_per_epoch: int) -> optax.Schedule:
    total_steps = config.epochs * steps_per_epoch
    warmup_steps = max(1, int(total_steps * config.warmup_fraction))
    cosine_steps = max(1, total_steps - warmup_steps)
    cosine = optax.cosine_decay_schedule(config.learning_rate, cosine_steps, alpha=0.01)
    return optax.join_schedules(
        schedules=[
            optax.linear_schedule(0.0, config.learning_rate, warmup_steps),
            cosine,
        ],
        boundaries=[warmup_steps],
    )


# ============================================================
# 5.  Training
# ============================================================

def main():
    config = HNOConfig()
    os.makedirs(config.results_dir, exist_ok=True)
    os.makedirs(config.ckpt_dir, exist_ok=True)

    print("=" * 60)
    print("HNO H1-Autograd Training  (30d data only)")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 5a. Discover & Load data
    # ------------------------------------------------------------------
    train_files, test_files = discover_files(config)
    print(f"\nTrain files ({len(train_files)}):")
    for f in train_files: print(f"  {os.path.basename(f)}")
    print(f"Test  files ({len(test_files)}):")
    for f in test_files:  print(f"  {os.path.basename(f)}")

    print("\n--- Loading train data ---")
    x_train, y_train, theta_train, Br_train, Bt_train = load_dataset(train_files, config)
    print(f"  x_train={x_train.shape}  y_train={y_train.shape}")

    print("--- Loading test data ---")
    x_test,  y_test,  theta_test,  Br_test,  Bt_test  = load_dataset(test_files,  config)
    print(f"  x_test={x_test.shape}  y_test={y_test.shape}")

    n_train = x_train.shape[0]

    # ------------------------------------------------------------------
    # 5b. Normalisation
    # ------------------------------------------------------------------
    x_norm = UnitGaussianNormalizer(x_train)
    y_norm = UnitGaussianNormalizer(y_train)

    x_tr_enc = jnp.asarray(x_norm.encode(x_train))
    y_tr_enc = jnp.asarray(y_norm.encode(y_train))
    x_te_enc = jnp.asarray(x_norm.encode(x_test))

    # Move heavy arrays to jnp once
    # We'll use y_norm.mean and y_norm.std directly for the physical decoding
    y_std = jnp.asarray(y_norm.std, dtype=jnp.float32)
    y_mean = jnp.asarray(y_norm.mean, dtype=jnp.float32)

    # Move heavy arrays to jnp once
    y_tr_raw  = jnp.asarray(y_train)
    theta_tr  = jnp.asarray(theta_train)
    # Br here will represent the squared magnitude |B|^2 for the updated loss
    B2_tr     = jnp.asarray(Br_train**2 + Bt_train**2) # If Br/Bt are proxies, this is |B|^2
    # Actually, B_mag_grid was used to create Br_train/Bt_train as |B|/sqrt(2).
    # So (B/sqrt(2))^2 + (B/sqrt(2))^2 = B^2. Correct.

    y_te_raw  = jnp.asarray(y_test)
    theta_te  = jnp.asarray(theta_test)

    # Save normalizers
    norm_path = os.path.join(config.ckpt_dir, "normalizers.pkl")
    with open(norm_path, "wb") as f:
        pickle.dump({"x_norm": x_norm, "y_norm": y_norm}, f)
    print(f"\nNormalizers saved → {norm_path}")

    # ------------------------------------------------------------------
    # 5c. Model
    # ------------------------------------------------------------------
    model = HNO(
        hidden_channels=config.hidden_channels,
        n_layers=config.n_layers,
        n_modes=config.n_modes,
        out_channels=config.out_channels,
        lifting_channel_ratio=config.lifting_channel_ratio,
        projection_channel_ratio=config.projection_channel_ratio,
        use_channel_mlp=config.use_channel_mlp,
        channel_mlp_skip=config.channel_mlp_skip,
        hno_skip=config.hno_skip,
        use_norm=config.use_norm,
        use_grid=False,     # coordinates already in input channels
        padding=config.domain_padding,
        activation=nn.gelu,
        order=0,            # Bessel J0
        r_axis=1,
        theta_axis=2,
    )

    rng = jax.random.PRNGKey(config.seed)
    rng, init_rng = jax.random.split(rng)

    steps_per_epoch = max(1, n_train // config.batch_size)
    schedule = make_schedule(config, steps_per_epoch)

    input_shape = (config.batch_size, config.res_r, config.res_theta, x_train.shape[-1])
    state = create_train_state(
        rng=init_rng,
        model=model,
        input_shape=input_shape,
        learning_rate=schedule,
        weight_decay=config.weight_decay,
    )

    # ------------------------------------------------------------------
    # 5d. JIT train step with H1-autograd loss
    # ------------------------------------------------------------------
    @jax.jit
    def train_step(state, batch, rng_key):
        def loss_fn(params):
            # Wrap apply_fn so its output is in *physical* units (decoded Az).
            def decoded_apply(variables, x_enc):
                pred_enc = state.apply_fn(variables, x_enc)
                return pred_enc * y_std + y_mean

            loss = h1_loss_hno_autograd_subsampled(
                apply_fn=decoded_apply,
                params=params,
                x_batch=batch["x"],
                target=batch["y_raw"],
                theta_grid=batch["theta"],
                B_r_true=jnp.sqrt(batch["B2"]), # We pass B_mag to be squared inside
                B_theta_true=jnp.zeros_like(batch["B2"]), # Not used in new formula
                rng=rng_key,
                subsample_n=config.subsample_n,
                r_ch_idx=config.r_ch_idx,           # channel index of r in x_enc
                theta_ch_idx=config.theta_ch_idx,   # channel index of theta in x_enc
                data_weight=config.data_weight,
                deriv_weight=config.deriv_weight,
            )
            return loss

        grad_fn = jax.value_and_grad(loss_fn)
        loss, grads = grad_fn(state.params)
        state = state.apply_gradients(grads=grads)
        return state, loss

    # ------------------------------------------------------------------
    # 5e. Training loop
    # ------------------------------------------------------------------
    print(f"\nStarting training: {config.epochs} epochs, "
          f"batch={config.batch_size}, subsample_n={config.subsample_n}, "
          f"deriv_weight={config.deriv_weight}")

    history = {"train_loss": [], "train_data_l2": [], "test_rel_l2": []}
    best_test_l2 = float("inf")
    ckpt_path = os.path.join(config.ckpt_dir, "best_model.pkl")
    train_rng = jax.random.PRNGKey(config.seed + 1)
    t_total = time.time()

    for epoch in range(config.epochs):
        t_epoch = time.time()
        epoch_loss = 0.0

        # shuffle
        train_rng, shuf_rng = jax.random.split(train_rng)
        perm = np.array(jax.random.permutation(shuf_rng, n_train))

        for step in range(steps_per_epoch):
            train_rng, sub_rng = jax.random.split(train_rng)
            idx = perm[step * config.batch_size : (step + 1) * config.batch_size]
            if len(idx) < config.batch_size:
                continue

            batch = {
                "x":     x_tr_enc[idx],
                "y_raw": y_tr_raw[idx],
                "theta": theta_tr[idx],
                "B2":    B2_tr[idx],
            }
            state, loss = train_step(state, batch, sub_rng)
            epoch_loss += float(loss)

        avg_loss = epoch_loss / steps_per_epoch
        history["train_loss"].append(avg_loss)

        # ---- pure data L2 on train (no deriv term) ----
        tr_pred_enc = state.apply_fn({"params": state.params}, x_tr_enc)
        tr_pred     = y_norm.decode(tr_pred_enc)
        tr_l2       = float(lp_loss(tr_pred, y_train))
        history["train_data_l2"].append(tr_l2)

        # ---- test rel L2 ----
        te_pred_enc = state.apply_fn({"params": state.params}, x_te_enc)
        te_pred     = y_norm.decode(te_pred_enc)
        te_l2       = float(lp_loss(te_pred, y_test))
        history["test_rel_l2"].append(te_l2)

        if te_l2 < best_test_l2:
            best_test_l2 = te_l2
            with open(ckpt_path, "wb") as f:
                f.write(flax.serialization.to_bytes(state.params))

        if epoch % 5 == 0 or epoch == config.epochs - 1:
            current_lr = float(schedule(state.step))
            elapsed = time.time() - t_epoch
            print(
                f"Epoch {epoch:4d}/{config.epochs} | "
                f"H1 Loss: {avg_loss:.4e} | "
                f"Train L2: {tr_l2:.4f} | "
                f"Test  L2: {te_l2:.4f} | "
                f"Best: {best_test_l2:.4f} | "
                f"LR: {current_lr:.2e} | "
                f"t: {elapsed:.1f}s"
            )

    print(f"\nTraining complete in {(time.time()-t_total)/60:.1f} min. "
          f"Best Test Rel L2: {best_test_l2:.6f}")

    # ------------------------------------------------------------------
    # 5f. Save history
    # ------------------------------------------------------------------
    hist_path = os.path.join(config.results_dir, "history.json")
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"History → {hist_path}")

    # ------------------------------------------------------------------
    # 5g. Loss curves
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].semilogy(history["train_loss"],    label="H1 train loss")
    axes[0].semilogy(history["train_data_l2"], label="Train data L2", linestyle="--")
    axes[0].set_title("Training Loss"); axes[0].legend(); axes[0].grid(True)
    axes[1].semilogy(history["test_rel_l2"],   label="Test Rel L2",   color="tab:orange")
    axes[1].set_title("Test Relative L2 Error"); axes[1].legend(); axes[1].grid(True)
    fig.suptitle("HNO – H1 Autograd Training (30d)")
    curve_path = os.path.join(config.results_dir, "training_curves.png")
    plt.tight_layout(); plt.savefig(curve_path, dpi=150)
    print(f"Loss curves → {curve_path}")

    # ------------------------------------------------------------------
    # 5h. Prediction plots for test samples
    # ------------------------------------------------------------------
    print("Generating prediction plots...")
    with open(ckpt_path, "rb") as f:
        best_params = flax.serialization.from_bytes(state.params, f.read())

    te_pred_best_enc = state.apply_fn({"params": best_params}, x_te_enc)
    te_pred_best     = np.array(y_norm.decode(te_pred_best_enc))

    for idx in range(min(len(test_files), 2)):
        name = os.path.basename(test_files[idx]).replace("_Az_30d_uniform.csv", "")
        
        # Pull coordinates from raw x_test [batch, R, Theta, Channel]
        # Channels: 0=theta, 1=r
        theta_grid = x_test[idx, ..., 0]
        r_grid     = x_test[idx, ..., 1]
        
        # Convert to Cartesian for plotting
        X = r_grid * np.cos(theta_grid)
        Y = r_grid * np.sin(theta_grid)
        
        true_field = np.array(y_test)[idx, ..., 0]
        pred_field = te_pred_best[idx, ..., 0]
        diff_field = np.abs(true_field - pred_field)

        fig, axs = plt.subplots(1, 3, figsize=(18, 5))
        vmin, vmax = true_field.min(), true_field.max()
        
        # Ground Truth
        im0 = axs[0].pcolormesh(X, Y, true_field, cmap="viridis", vmin=vmin, vmax=vmax, shading="gouraud")
        axs[0].set_title("Ground Truth Az (Cartesian)"); plt.colorbar(im0, ax=axs[0])
        axs[0].set_aspect("equal")
        
        # Prediction
        im1 = axs[1].pcolormesh(X, Y, pred_field, cmap="viridis", vmin=vmin, vmax=vmax, shading="gouraud")
        axs[1].set_title("HNO Prediction (Cartesian)");  plt.colorbar(im1, ax=axs[1])
        axs[1].set_aspect("equal")
        
        # Difference
        im2 = axs[2].pcolormesh(X, Y, diff_field, cmap="magma", shading="gouraud")
        axs[2].set_title("|Diff|"); plt.colorbar(im2, ax=axs[2])
        axs[2].set_aspect("equal")
        
        fig.suptitle(f"Test: {name}")
        plt.tight_layout()
        plot_path = os.path.join(config.results_dir, f"pred_{idx}_{name}.png")
        plt.savefig(plot_path, dpi=150)
        print(f"  Saved → {plot_path}")
        plt.close(fig)

    print("\nAll done.")


if __name__ == "__main__":
    main()
