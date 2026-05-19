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

"""
Benchmark FNO: PyTorch (neuralop) vs JAX (SciREX).

Trains matched FNO models on the same 2D Poisson dataset and produces:
  1. Train Loss vs Epoch (both frameworks)
  2. Test L2 Error vs Epoch (both frameworks)
  3. Time per Epoch vs Epoch (both frameworks)

Usage
-----
    python scripts/benchmark_fno_pytorch_vs_jax.py [--epochs N] [--quick]

    --epochs N   : number of training epochs (default: 100)
    --quick      : use small data / model for a fast smoke-test
"""

import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
import sys
import time
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── project root on path ─────────────────────────────────────────────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
NEURALOP_ROOT = os.path.join(PROJECT_ROOT, "temp_no_repo")
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if NEURALOP_ROOT not in sys.path:
    sys.path.insert(0, NEURALOP_ROOT)

# ── output dirs ──────────────────────────────────────────────────────────────
RESULTS_DIR = os.path.join(PROJECT_ROOT, "experiments", "results", "fno_benchmark")
os.makedirs(RESULTS_DIR, exist_ok=True)

# =============================================================================
# Shared Hyper-parameters
# =============================================================================
SEED        = 42
N_TRAIN     = 1000   # number of training samples
N_TEST      = 200    # number of test samples
BATCH_SIZE  = 20     # mini-batch size
NX, NY      = 64, 64 # spatial resolution
N_MODES     = (16, 16)
HIDDEN_CH   = 64
N_LAYERS    = 4
LR          = 1e-3
WEIGHT_DECAY = 1e-4


# =============================================================================
# Shared Data Generation  (pure NumPy – framework-agnostic)
# =============================================================================

def _grf_field(nx, ny, alpha=2.0, tau=3.0, rng=None):
    """Draw one 2-D Gaussian Random Field sample."""
    kx = np.fft.fftfreq(nx) * nx
    ky = np.fft.fftfreq(ny) * ny
    Kx, Ky = np.meshgrid(kx, ky, indexing="ij")
    k_sq = Kx**2 + Ky**2
    inv_eigen = 1.0 / ((k_sq + tau**2) ** alpha)
    inv_eigen[0, 0] = 0.0
    noise = rng.normal(size=(nx, ny)) + 1j * rng.normal(size=(nx, ny))
    F_hat = noise * inv_eigen * nx * ny
    field = np.fft.ifft2(F_hat).real
    std = field.std()
    if std > 0:
        field /= std
    return field.astype(np.float32)


def _poisson_solve(f_2d):
    """Solve -∇²u = f on periodic domain via FFT."""
    nx, ny = f_2d.shape
    kx = np.fft.fftfreq(nx, d=1.0 / nx) * 2.0 * np.pi
    ky = np.fft.fftfreq(ny, d=1.0 / ny) * 2.0 * np.pi
    Kx, Ky = np.meshgrid(kx, ky, indexing="ij")
    k2 = Kx**2 + Ky**2
    k2[0, 0] = 1.0
    F_hat = np.fft.fft2(f_2d)
    U_hat = F_hat / k2
    U_hat[0, 0] = 0.0
    return np.fft.ifft2(U_hat).real.astype(np.float32)


def generate_poisson_data(n_samples, nx, ny, seed):
    """
    Returns
    -------
    f_raw : (N, nx, ny, 1)   – source field only
    f_grid : (N, nx, ny, 3)  – source + x-grid + y-grid  (JAX / channels-last)
    u      : (N, nx, ny, 1)  – solution field
    """
    rng = np.random.default_rng(seed)
    xs = np.linspace(0, 1, nx, dtype=np.float32)
    ys = np.linspace(0, 1, ny, dtype=np.float32)
    Xg, Yg = np.meshgrid(xs, ys, indexing="ij")

    f_raw  = np.zeros((n_samples, nx, ny, 1), dtype=np.float32)
    u      = np.zeros((n_samples, nx, ny, 1), dtype=np.float32)

    for i in range(n_samples):
        fi = _grf_field(nx, ny, rng=rng)
        f_raw[i, :, :, 0] = fi
        u[i, :, :, 0]     = _poisson_solve(fi)

    grid_x = np.broadcast_to(Xg[None, :, :, None], (n_samples, nx, ny, 1)).copy()
    grid_y = np.broadcast_to(Yg[None, :, :, None], (n_samples, nx, ny, 1)).copy()
    f_grid = np.concatenate([f_raw, grid_x, grid_y], axis=-1)  # (N,nx,ny,3)

    return f_raw, f_grid, u


# =============================================================================
# Normalizer (shared logic)
# =============================================================================

class UnitGaussianNorm:
    """Zero-mean, unit-variance normalizer (computed on the training set)."""

    def __init__(self, data):
        self.mean = float(data.mean())
        self.std  = float(data.std()) + 1e-8

    def encode(self, x):
        return (x - self.mean) / self.std

    def decode(self, x):
        return x * self.std + self.mean


# =============================================================================
# Relative L2 loss (numpy)
# =============================================================================

def rel_l2_numpy(pred, target):
    """Relative L2 error averaged over batch."""
    diff   = pred - target
    numer  = np.sqrt((diff**2).sum(axis=(1, 2, 3)))
    denom  = np.sqrt((target**2).sum(axis=(1, 2, 3))) + 1e-8
    return float((numer / denom).mean())


# =============================================================================
#  P Y T O R C H   training
# =============================================================================

def train_pytorch(f_grid_np, u_np, f_test_np, u_test_np,
                  x_norm, y_norm, epochs, batch_size):
    """
    Train the neuralop PyTorch FNO and return metrics dicts.

    Parameters
    ----------
    f_grid_np : (N, nx, ny, 3)  numpy, channels-last
    u_np      : (N, nx, ny, 1)  numpy
    Returns lists: train_losses, test_losses, epoch_times
    """
    import torch
    import torch.nn as nn

    # ── import neuralop FNO ───────────────────────────────────────────────
    from neuralop.models import FNO
    from neuralop.losses import LpLoss

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        try:
            # Test model execution to verify cuDNN/CUDA capability (Blackwell compatibility)
            test_model = FNO(
                n_modes=(4, 4),
                in_channels=3,
                out_channels=1,
                hidden_channels=8,
                n_layers=1,
                lifting_channel_ratio=1,
                projection_channel_ratio=1,
                positional_embedding=None,
            ).cuda()
            with torch.no_grad():
                test_model(torch.randn(1, 3, 16, 16).cuda())
        except RuntimeError as e:
            print(f"[PyTorch] Warning: CUDA is available but incompatible with this PyTorch build ({e}). Falling back to CPU.")
            device = torch.device("cpu")
    print(f"[PyTorch] device = {device}")

    # neuralop FNO expects channels-first: (N, C, H, W)
    # f_grid_np is (N, nx, ny, 3)
    f_enc = x_norm.encode(f_grid_np)
    u_enc = y_norm.encode(u_np)

    # channels-first conversion
    f_tr = torch.tensor(np.transpose(f_enc, (0, 3, 1, 2)), dtype=torch.float32)
    u_tr = torch.tensor(np.transpose(u_enc, (0, 3, 1, 2)), dtype=torch.float32)
    f_te = torch.tensor(np.transpose(x_norm.encode(f_test_np), (0, 3, 1, 2)), dtype=torch.float32)
    u_te_raw = torch.tensor(np.transpose(u_test_np, (0, 3, 1, 2)), dtype=torch.float32)

    # PyTorch dataset / loader
    dataset  = torch.utils.data.TensorDataset(f_tr, u_tr)
    loader   = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Model: in_channels=3 (f + grid_x + grid_y), positional_embedding=None
    model = FNO(
        n_modes=N_MODES,
        in_channels=3,
        out_channels=1,
        hidden_channels=HIDDEN_CH,
        n_layers=N_LAYERS,
        lifting_channel_ratio=2,
        projection_channel_ratio=2,
        positional_embedding=None,   # grid already baked in
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[PyTorch] model params = {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    lp_loss   = LpLoss(d=2, p=2, reduction="sum")

    train_losses, test_losses, epoch_times = [], [], []

    print(f"[PyTorch] Starting {epochs} epochs …")
    for epoch in range(epochs):
        t0 = time.perf_counter()
        model.train()
        epoch_loss = 0.0
        n_batches  = 0

        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = lp_loss(pred, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches  += 1

        scheduler.step()
        epoch_time = time.perf_counter() - t0

        # ── test evaluation ───────────────────────────────────────────────
        model.eval()
        with torch.no_grad():
            pred_enc = model(f_te.to(device)).cpu().numpy()          # (N,1,nx,ny)
            pred_enc = np.transpose(pred_enc, (0, 2, 3, 1))          # → (N,nx,ny,1)
            pred_dec = y_norm.decode(pred_enc)
            test_l2  = rel_l2_numpy(pred_dec, u_test_np)

        if epoch == 0:
            print(f"[PT DBG] Epoch 0 pred_dec mean={pred_dec.mean():.4f} std={pred_dec.std():.4f} min={pred_dec.min():.4f} max={pred_dec.max():.4f}")
            print(f"[PT DBG] Epoch 0 u_test_np mean={u_test_np.mean():.4f} std={u_test_np.std():.4f} min={u_test_np.min():.4f} max={u_test_np.max():.4f}")

        avg_train = epoch_loss / max(n_batches, 1)
        train_losses.append(avg_train)
        test_losses.append(test_l2)
        epoch_times.append(epoch_time)

        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"  [PT] epoch {epoch:4d} | train={avg_train:.4e} "
                  f"| test_l2={test_l2:.4f} | time={epoch_time:.2f}s")

    return train_losses, test_losses, epoch_times


# =============================================================================
#  J A X   training
# =============================================================================

def train_jax(f_grid_np, u_np, f_test_np, u_test_np,
              x_norm, y_norm, epochs, batch_size, pt_model=None):
    """
    Train the SciREX JAX FNO and return metrics dicts.
    """
    # Set XLA determinism flags before importing JAX
    os.environ["XLA_FLAGS"] = (
        os.environ.get("XLA_FLAGS", "") + " --xla_gpu_deterministic_ops=true"
    )
    os.environ["TF_CUDNN_DETERMINISTIC"] = "1"

    import jax
    import jax.numpy as jnp
    import optax
    from flax import linen as nn

    from scirex.operators.models.fno import FNO
    from scirex.operators.training import create_train_state, UnitGaussianNormalizer
    from scirex.operators.losses import lp_loss

    print(f"[JAX] devices = {jax.devices()}")

    # ── encode ────────────────────────────────────────────────────────────
    n_train = f_grid_np.shape[0]
    f_enc    = x_norm.encode(f_grid_np).astype(np.float32)
    u_enc    = y_norm.encode(u_np).astype(np.float32)
    f_te_enc = x_norm.encode(f_test_np).astype(np.float32)

    f_enc    = jnp.asarray(f_enc)
    u_enc    = jnp.asarray(u_enc)
    f_te_enc = jnp.asarray(f_te_enc)
    u_te_jnp = jnp.asarray(u_test_np)

    # ── model + state ─────────────────────────────────────────────────────
    model = FNO(
        hidden_channels=HIDDEN_CH,
        n_layers=N_LAYERS,
        n_modes=N_MODES,
        out_channels=1,
        lifting_channel_ratio=2,
        projection_channel_ratio=2,
        use_grid=False,  # Match PyTorch's positional_embedding=None (already in input)
        use_norm=False,
        fno_skip="linear",
        channel_mlp_skip="soft-gating",
        use_channel_mlp=True,
        padding=0.0,
        activation=nn.gelu,
    )

    steps_per_epoch = n_train // batch_size

    jax_lr = 5e-3
    # Staircase decay: drops learning rate by 0.5 every 50 epochs (matching PyTorch's StepLR(50, 0.5) exactly)
    schedule = optax.exponential_decay(
        init_value=jax_lr,
        decay_rate=0.5,
        transition_steps=50 * steps_per_epoch,
        staircase=True,
    )

    # Robust AdamW optimizer to match PyTorch's convergence dynamics
    tx = optax.adamw(learning_rate=schedule, weight_decay=WEIGHT_DECAY)

    rng = jax.random.PRNGKey(SEED)
    input_shape = (batch_size, NX, NY, 3)
    state = create_train_state(
        rng=rng,
        model=model,
        input_shape=input_shape,
        tx=tx,
    )

    # Manually map PyTorch weights to JAX train state!
    if pt_model is not None:
        def to_jax_dense(pt_weight):
            w = pt_weight.detach().numpy()
            if w.ndim == 3: # Conv1d
                w = w[:, :, 0]
            return np.transpose(w)
            
        def to_jax_bias(pt_bias):
            return pt_bias.detach().numpy()
            
        print("[JAX] Overriding initialization with PyTorch's initial weights!")
        pt_dict = pt_model.state_dict()
        new_params = jax.tree_util.tree_map(lambda x: x, state.params)
        
        new_params['ChannelMLP_0']['dense_0']['kernel'] = jnp.array(to_jax_dense(pt_dict['lifting.fcs.0.weight']))
        new_params['ChannelMLP_0']['dense_0']['bias'] = jnp.array(to_jax_bias(pt_dict['lifting.fcs.0.bias']))
        new_params['ChannelMLP_0']['dense_1']['kernel'] = jnp.array(to_jax_dense(pt_dict['lifting.fcs.1.weight']))
        new_params['ChannelMLP_0']['dense_1']['bias'] = jnp.array(to_jax_bias(pt_dict['lifting.fcs.1.bias']))
        
        new_params['ChannelMLP_1']['dense_0']['kernel'] = jnp.array(to_jax_dense(pt_dict['projection.fcs.0.weight']))
        new_params['ChannelMLP_1']['dense_0']['bias'] = jnp.array(to_jax_bias(pt_dict['projection.fcs.0.bias']))
        new_params['ChannelMLP_1']['dense_1']['kernel'] = jnp.array(to_jax_dense(pt_dict['projection.fcs.1.weight']))
        new_params['ChannelMLP_1']['dense_1']['bias'] = jnp.array(to_jax_bias(pt_dict['projection.fcs.1.bias']))
        
        for i in range(4):
            block_jax = f'FNOBlock_{i}'
            new_params[block_jax]['SkipConnection_0']['Dense_0']['kernel'] = jnp.array(to_jax_dense(pt_dict[f'fno_blocks.fno_skips.{i}.conv.weight']))
            if pt_model.fno_blocks.fno_skips[i].conv.bias is not None:
                new_params[block_jax]['SkipConnection_0']['Dense_0']['bias'] = jnp.array(to_jax_bias(pt_dict[f'fno_blocks.fno_skips.{i}.conv.bias']))
                
            pt_w = pt_dict[f'fno_blocks.convs.{i}.weight.tensor'].detach().numpy()
            w1 = pt_w[:, :, :8, :]
            w2 = pt_w[:, :, -8:, :]
            new_params[block_jax]['SpectralConv_0']['weights_r_1'] = jnp.array(np.real(w1))
            new_params[block_jax]['SpectralConv_0']['weights_i_1'] = jnp.array(np.imag(w1))
            new_params[block_jax]['SpectralConv_0']['weights_r_2'] = jnp.array(np.real(w2))
            new_params[block_jax]['SpectralConv_0']['weights_i_2'] = jnp.array(np.imag(w2))
            if pt_model.fno_blocks.convs[i].bias is not None:
                new_params[block_jax]['SpectralConv_0']['bias'] = jnp.array(pt_dict[f'fno_blocks.convs.{i}.bias'].detach().numpy()[:, 0, 0])
                
            new_params[block_jax]['ChannelMLP_0']['dense_0']['kernel'] = jnp.array(to_jax_dense(pt_dict[f'fno_blocks.channel_mlp.{i}.fcs.0.weight']))
            new_params[block_jax]['ChannelMLP_0']['dense_0']['bias'] = jnp.array(to_jax_bias(pt_dict[f'fno_blocks.channel_mlp.{i}.fcs.0.bias']))
            new_params[block_jax]['ChannelMLP_0']['dense_1']['kernel'] = jnp.array(to_jax_dense(pt_dict[f'fno_blocks.channel_mlp.{i}.fcs.1.weight']))
            new_params[block_jax]['ChannelMLP_0']['dense_1']['bias'] = jnp.array(to_jax_bias(pt_dict[f'fno_blocks.channel_mlp.{i}.fcs.1.bias']))
            
            sg_w = pt_dict[f'fno_blocks.channel_mlp_skips.{i}.weight'].detach().numpy()
            new_params[block_jax]['SkipConnection_1']['SoftGating_0']['weight'] = jnp.array(np.transpose(sg_w, (0, 2, 3, 1)))

        state = state.replace(params=new_params)

    n_params = sum(x.size for x in jax.tree_util.tree_leaves(state.params))
    print(f"[JAX] model params = {n_params:,}")

    steps_per_epoch = n_train // batch_size

    @jax.jit
    def train_step(state, batch):
        def loss_fn(params):
            pred = state.apply_fn({"params": params}, batch["x"])
            diff = (pred - batch["y"]).reshape(batch_size, -1)
            targ = batch["y"].reshape(batch_size, -1)
            diff_norm = jnp.linalg.norm(diff, ord=2, axis=1)
            targ_norm = jnp.linalg.norm(targ, ord=2, axis=1)
            return jnp.mean(diff_norm / (targ_norm + 1e-8))
        grad_fn = jax.value_and_grad(loss_fn)
        loss, grads = grad_fn(state.params)
        state = state.apply_gradients(grads=grads)
        return state, loss

    f_enc_jnp = jnp.array(f_enc)
    u_enc_jnp = jnp.array(u_enc)

    train_losses, test_losses, epoch_times = [], [], []
    rng_key = jax.random.PRNGKey(SEED + 1)

    print(f"[JAX] Starting {epochs} epochs …")
    for epoch in range(epochs):
        t0 = time.perf_counter()

        rng_key, sk = jax.random.split(rng_key)
        perm       = jax.random.permutation(sk, n_train)
        f_shuf     = f_enc_jnp[perm]
        u_shuf     = u_enc_jnp[perm]

        epoch_loss = 0.0
        for step in range(steps_per_epoch):
            s = step * batch_size
            e = s + batch_size
            batch = {"x": f_shuf[s:e], "y": u_shuf[s:e]}
            state, loss = train_step(state, batch)
            epoch_loss += float(loss)

        epoch_time = time.perf_counter() - t0
        avg_train  = epoch_loss / steps_per_epoch

        # ── test ──────────────────────────────────────────────────────────
        pred_enc = state.apply_fn({"params": state.params}, f_te_enc)
        pred_dec = jnp.array(y_norm.decode(np.array(pred_enc)))
        test_l2  = float(lp_loss(pred_dec, u_te_jnp))

        if epoch == 0:
            print(f"[JAX DBG] Epoch 0 pred_dec mean={pred_dec.mean():.4f} std={pred_dec.std():.4f} min={pred_dec.min():.4f} max={pred_dec.max():.4f}")
            print(f"[JAX DBG] Epoch 0 u_te_jnp mean={u_te_jnp.mean():.4f} std={u_te_jnp.std():.4f} min={u_te_jnp.min():.4f} max={u_te_jnp.max():.4f}")

        train_losses.append(avg_train)
        test_losses.append(test_l2)
        epoch_times.append(epoch_time)

        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"  [JAX] epoch {epoch:4d} | train={avg_train:.4e} "
                  f"| test_l2={test_l2:.4f} | time={epoch_time:.2f}s")

    return train_losses, test_losses, epoch_times


# =============================================================================
#  P L O T T I N G
# =============================================================================

COLORS = {
    "pytorch": "#2563EB",   # blue
    "jax":     "#DC2626",   # red
}
STYLE = {
    "pytorch": dict(color=COLORS["pytorch"], marker="o", markersize=4,
                    linestyle="-",  linewidth=2.0),
    "jax":     dict(color=COLORS["jax"],     marker="s", markersize=4,
                    linestyle="--", linewidth=2.0),
}


def plot_loss_comparison(epochs_range, pt_train, pt_test, jax_train, jax_test, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("FNO Training Dynamics: PyTorch vs JAX", fontsize=14, fontweight="bold")

    for ax, (pt_vals, jax_vals, title, ylabel) in zip(
        axes,
        [
            (pt_train,  jax_train,  "Training Loss vs Epochs",   "Average Training Loss"),
            (pt_test,   jax_test,   "Test L2 Error vs Epochs",   "Test L2 Error"),
        ],
    ):
        ax.plot(epochs_range, pt_vals,  label="FNO PyTorch (neuralop)", **STYLE["pytorch"])
        ax.plot(epochs_range, jax_vals, label="FNO JAX (SciREX)",       **STYLE["jax"])

        # shaded area between the two
        ax.fill_between(
            epochs_range,
            np.minimum(pt_vals, jax_vals),
            np.maximum(pt_vals, jax_vals),
            alpha=0.12,
            color="#F97316",
        )

        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.legend(framealpha=0.9)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.set_xlim(left=0)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] Saved: {out_path}")


def plot_time_comparison(epochs_range, pt_times, jax_times, out_path):
    avg_pt  = float(np.mean(pt_times))
    avg_jax = float(np.mean(jax_times))
    speedup = avg_pt / (avg_jax + 1e-9)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_title("Training Performance: FNO Implementation Comparison",
                 fontsize=13, fontweight="bold")

    ax.plot(epochs_range, pt_times,  label="PyTorch (neuralop)", **STYLE["pytorch"])
    ax.plot(epochs_range, jax_times, label="JAX (SciREX)",        **STYLE["jax"])

    # annotation box
    info = (f"Avg PyTorch: {avg_pt:.2f}s\n"
            f"Avg JAX:     {avg_jax:.2f}s\n"
            f"Speedup:     {speedup:.2f}x")
    ax.text(
        0.97, 0.97, info,
        transform=ax.transAxes, fontsize=10,
        verticalalignment="top", horizontalalignment="right",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#9CA3AF", alpha=0.9),
    )

    # mark first JAX epoch as compilation overhead if it stands out
    if len(jax_times) > 1 and jax_times[0] > 2 * np.median(jax_times[1:]):
        ax.annotate(
            "JAX Compilation\nOverhead",
            xy=(0, jax_times[0]),
            xytext=(max(1, len(epochs_range) // 8), jax_times[0] * 0.92),
            arrowprops=dict(arrowstyle="->", color="black"),
            fontsize=9,
        )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Time per Epoch (seconds)")
    ax.legend(framealpha=0.9)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] Saved: {out_path}")


# =============================================================================
#  M A I N
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Benchmark FNO: PyTorch vs JAX")
    p.add_argument("--epochs", type=int, default=100,
                   help="Number of training epochs (default: 100)")
    p.add_argument("--quick",  action="store_true",
                   help="Quick smoke-test (20 epochs, tiny data)")
    p.add_argument("--skip-pytorch", action="store_true",
                   help="Skip PyTorch training (load cached results if available)")
    p.add_argument("--skip-jax",     action="store_true",
                   help="Skip JAX training (load cached results if available)")
    return p.parse_args()


def load_or_train(cache_path, train_fn, skip_flag, label):
    if skip_flag:
        if os.path.exists(cache_path):
            print(f"[{label}] Loading cached results from {cache_path}")
            with open(cache_path) as f:
                d = json.load(f)
            return d["train"], d["test"], d["times"]
        else:
            print(f"[{label}] Skip requested but no cache found. Returning empty metrics.")
            return [], [], []
    results = train_fn()
    with open(cache_path, "w") as f:
        json.dump({"train": results[0], "test": results[1], "times": results[2]}, f, indent=2)
    return results


def main():
    args = parse_args()

    epochs = 20 if args.quick else args.epochs
    n_tr   = 200 if args.quick else N_TRAIN
    n_te   = 40  if args.quick else N_TEST
    bsz    = 20  if args.quick else BATCH_SIZE

    print("=" * 60)
    print(f"FNO Benchmark  |  epochs={epochs}  n_train={n_tr}  n_test={n_te}")
    print("=" * 60)

    # ── Generate shared dataset ───────────────────────────────────────────
    print("\n[DATA] Generating Poisson 2D dataset …")
    _, f_grid_tr, u_tr = generate_poisson_data(n_tr, NX, NY, seed=SEED)
    _, f_grid_te, u_te = generate_poisson_data(n_te, NX, NY, seed=999)

    # Fit normalizers on training data only
    x_norm = UnitGaussianNorm(f_grid_tr)
    y_norm = UnitGaussianNorm(u_tr)
    print(f"[DATA] Train: f={f_grid_tr.shape}  u={u_tr.shape}")
    print(f"[DATA] Test : f={f_grid_te.shape}  u={u_te.shape}")

    # ── PyTorch ───────────────────────────────────────────────────────────
    pt_cache = os.path.join(RESULTS_DIR, "pytorch_metrics.json")

    def run_pytorch():
        print("\n" + "=" * 40)
        print("  Training PyTorch FNO …")
        print("=" * 40)
        return train_pytorch(f_grid_tr, u_tr, f_grid_te, u_te,
                             x_norm, y_norm, epochs, bsz)

    pt_train, pt_test, pt_times = load_or_train(
        pt_cache, run_pytorch, args.skip_pytorch, "PyTorch"
    )

    # ── JAX ───────────────────────────────────────────────────────────────
    jax_cache = os.path.join(RESULTS_DIR, "jax_metrics.json")

    def run_jax():
        print("\n" + "=" * 40)
        print("  Training JAX FNO …")
        print("=" * 40)
        return train_jax(f_grid_tr, u_tr, f_grid_te, u_te,
                         x_norm, y_norm, epochs, bsz)

    jax_train, jax_test, jax_times = load_or_train(
        jax_cache, run_jax, args.skip_jax, "JAX"
    )

    # ── Summary ──────────────────────────────────────────────────────────
    if (args.skip_pytorch and len(pt_train) == 0) or (args.skip_jax and len(jax_train) == 0):
        print("\n[INFO] Cross-environment run: intermediate phase completed.")
        return

    print("\n" + "=" * 60)
    print("  BENCHMARK SUMMARY")
    print("=" * 60)
    print(f"  PyTorch  | final train={pt_train[-1]:.4e}  "
          f"test={pt_test[-1]:.4f}  avg_time={np.mean(pt_times):.2f}s/epoch")
    print(f"  JAX      | final train={jax_train[-1]:.4e}  "
          f"test={jax_test[-1]:.4f}  avg_time={np.mean(jax_times):.2f}s/epoch")

    speedup = np.mean(pt_times) / (np.mean(jax_times) + 1e-9)
    print(f"\n  Speedup  (PyTorch / JAX avg epoch time) = {speedup:.2f}x")
    print("=" * 60)

    # Save combined metrics JSON
    combined = {
        "config": {
            "epochs": epochs, "n_train": n_tr, "n_test": n_te,
            "batch_size": bsz, "nx": NX, "ny": NY,
            "n_modes": list(N_MODES), "hidden_ch": HIDDEN_CH, "n_layers": N_LAYERS,
        },
        "pytorch": {"train": pt_train, "test": pt_test, "times": pt_times},
        "jax":     {"train": jax_train, "test": jax_test, "times": jax_times},
    }
    combined_path = os.path.join(RESULTS_DIR, "combined_metrics.json")
    with open(combined_path, "w") as f:
        json.dump(combined, f, indent=2)
    print(f"\n[SAVE] Metrics → {combined_path}")

    # ── Plots ─────────────────────────────────────────────────────────────
    # Align lengths for plotting in case one was cached with a different number of epochs
    plot_len = min(len(pt_train), len(jax_train))
    ep_range = list(range(1, plot_len + 1))

    plot_loss_comparison(
        ep_range, pt_train[:plot_len], pt_test[:plot_len], jax_train[:plot_len], jax_test[:plot_len],
        os.path.join(RESULTS_DIR, "fno_loss_comparison.png"),
    )
    plot_time_comparison(
        ep_range, pt_times[:plot_len], jax_times[:plot_len],
        os.path.join(RESULTS_DIR, "fno_time_comparison.png"),
    )

    print(f"\n[DONE] All results saved to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
