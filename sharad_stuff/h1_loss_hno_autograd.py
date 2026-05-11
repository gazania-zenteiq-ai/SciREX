# Copyright (c) 2024 Zenteiq Aitech Innovations Private Limited and
# AiREX Lab, Indian Institute of Science, Bangalore.
# All rights reserved.

import jax
import jax.numpy as jnp

def _diag_jac(fwd_fn, vals: jnp.ndarray) -> jnp.ndarray:
    """Diagonal of ∂fwd_fn(vals)/∂vals for a single sample.
    
    vals: (N,) or (N_r, N_theta)
    Returns: (..., C_out)
    """
    # jax.jacrev on a single sample (N) -> (N, C_out, N)
    full_jac = jax.jacrev(fwd_fn)(vals)
    
    # Identify output dimensions vs input dimension
    # vals.shape is (d1, d2, ...)
    # full_jac shape is (d1, d2, ..., C_out, d1, d2, ...)
    n_dim = vals.ndim
    out_spatial_shape = full_jac.shape[:n_dim]
    C_out = full_jac.shape[n_dim]
    N = vals.size
    
    # Flatten to (N, C_out, N)
    jac_flat = full_jac.reshape(N, C_out, N)
    # Extract diagonal over spatial axes (0 and 2)
    diag = jnp.diagonal(jac_flat.transpose(1, 0, 2), axis1=1, axis2=2)
    return diag.T.reshape(out_spatial_shape + (C_out,))

def h1_loss_hno_autograd(
    apply_fn,
    params,
    x_batch: jnp.ndarray,
    target: jnp.ndarray,
    theta_grid: jnp.ndarray,
    B_r_true: jnp.ndarray,
    B_theta_true: jnp.ndarray,
    r_ch_idx: int = 1,
    theta_ch_idx: int = 0,
    data_weight: float = 1.0,
    deriv_weight: float = 1.0,
    eps: float = 1e-8,
) -> jnp.ndarray:
    batch = x_batch.shape[0]
    pred = apply_fn({"params": params}, x_batch)

    # --- Data Loss ---
    pred_flat = pred.reshape(batch, -1)
    target_flat = target.reshape(batch, -1)
    rel_l2 = jnp.sqrt(jnp.sum((pred_flat - target_flat)**2, axis=-1)) / (jnp.sqrt(jnp.sum(target_flat**2, axis=-1)) + eps)

    # --- Derivative Loss (Batch Vmapt) ---
    def single_sample_derivs(x_one):
        r_vals = x_one[..., r_ch_idx]
        t_vals = x_one[..., theta_ch_idx]
        def f_r(r): return apply_fn({"params": params}, x_one.at[..., r_ch_idx].set(r)[None, ...])[0]
        def f_t(t): return apply_fn({"params": params}, x_one.at[..., theta_ch_idx].set(t)[None, ...])[0]
        return _diag_jac(f_r, r_vals), _diag_jac(f_t, t_vals)

    B_r_pred, B_t_pred = jax.vmap(single_sample_derivs)(x_batch)
    
    r_coords = x_batch[..., r_ch_idx][..., None]
    E_pred = B_r_pred**2 + (1.0 / (r_coords + eps))**2 * B_t_pred**2
    E_true = B_r_true**2 # |B|^2

    E_p_f = E_pred.reshape(batch, -1)
    E_t_f = E_true.reshape(batch, -1)
    rel_d = jnp.sqrt(jnp.sum((E_p_f - E_t_f)**2, axis=-1)) / (jnp.sqrt(jnp.sum(E_t_f**2, axis=-1)) + eps)

    return jnp.mean(data_weight * rel_l2 + deriv_weight * rel_d)

def h1_loss_hno_autograd_subsampled(
    apply_fn,
    params,
    x_batch: jnp.ndarray,
    target: jnp.ndarray,
    theta_grid: jnp.ndarray,
    B_r_true: jnp.ndarray,
    B_theta_true: jnp.ndarray,
    rng: jnp.ndarray,
    subsample_n: int = 512,
    r_ch_idx: int = 1,
    theta_ch_idx: int = 0,
    data_weight: float = 1.0,
    deriv_weight: float = 1.0,
    eps: float = 1e-8,
) -> jnp.ndarray:
    batch, NR, NT, C_in = x_batch.shape
    N = NR * NT
    pred = apply_fn({"params": params}, x_batch)

    # Full grid data loss
    pred_f = pred.reshape(batch, -1)
    targ_f = target.reshape(batch, -1)
    rel_l2 = jnp.sqrt(jnp.sum((pred_f - targ_f)**2, axis=-1)) / (jnp.sqrt(jnp.sum(targ_f**2, axis=-1)) + eps)

    # Subsample spatial points
    n = min(subsample_n, N)
    flat_idx = jax.random.choice(rng, N, shape=(n,), replace=False)
    
    x_sub = x_batch.reshape(batch, N, C_in)[:, flat_idx, :]
    E_true_sub = B_r_true.reshape(batch, N, -1)[:, flat_idx, :]**2 # |B|^2
    
    # mini-grid: (batch, n, 1, C_in)
    x_mini = x_sub.reshape(batch, n, 1, C_in)

    def single_sample_sub_derivs(xm):
        # xm: (n, 1, C_in)
        r_mini = xm[..., r_ch_idx] # (n, 1)
        t_mini = xm[..., theta_ch_idx] # (n, 1)
        def f_r(r): return apply_fn({"params": params}, xm.at[..., r_ch_idx].set(r)[None, ...])[0]
        def f_t(t): return apply_fn({"params": params}, xm.at[..., theta_ch_idx].set(t)[None, ...])[0]
        return _diag_jac(f_r, r_mini), _diag_jac(f_t, t_mini)

    Br_p_sub, Bt_p_sub = jax.vmap(single_sample_sub_derivs)(x_mini)
    
    r_sub = x_mini[..., r_ch_idx][..., None]
    E_p_sub = Br_p_sub**2 + (1.0 / (r_sub + eps))**2 * Bt_p_sub**2

    Ep_f = E_p_sub.reshape(batch, -1)
    Et_f = E_true_sub.reshape(batch, -1)
    rel_d = jnp.sqrt(jnp.sum((Ep_f - Et_f)**2, axis=-1)) / (jnp.sqrt(jnp.sum(Et_f**2, axis=-1)) + eps)

    return jnp.mean(data_weight * rel_l2 + deriv_weight * rel_d)
