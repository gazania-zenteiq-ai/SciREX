import numpy as np
import torch
import jax
import jax.numpy as jnp
from flax import linen as nn

# Import PyTorch neuralop model
from neuralop.models import FNO as PT_FNO
from neuralop.layers.spectral_convolution import SpectralConv as PT_SpectralConv

# Import JAX SciREX model
from scirex.operators.models.fno import FNO as JAX_FNO

def main():
    print("=== FNO Cross-Framework Verification ===")
    
    # 1. Initialize random input: (batch, nx, ny, channels)
    np.random.seed(42)
    x_np = np.random.randn(2, 64, 64, 3).astype(np.float32)
    
    # PyTorch expects channels-first: (batch, channels, nx, ny)
    x_pt = torch.tensor(np.transpose(x_np, (0, 3, 1, 2)))
    
    # JAX expects channels-last: (batch, nx, ny, channels)
    x_jax = jnp.array(x_np)
    
    # 2. Instantiate PyTorch FNO model
    pt_model = PT_FNO(
        n_modes=(16, 16),
        in_channels=3,
        out_channels=1,
        hidden_channels=64,
        n_layers=4,
        lifting_channel_ratio=2,
        projection_channel_ratio=2,
        positional_embedding=None,
        use_channel_mlp=True,
    )
    pt_model.eval()
    
    # 3. Instantiate JAX FNO model
    jax_model = JAX_FNO(
        hidden_channels=64,
        n_layers=4,
        n_modes=(16, 16),
        out_channels=1,
        lifting_channel_ratio=2,
        projection_channel_ratio=2,
        use_grid=False,
        use_norm=False,
        fno_skip="linear",
        channel_mlp_skip="soft-gating",
        use_channel_mlp=True,
        padding=0.0,
    )
    
    # Initialize JAX weights
    rng = jax.random.PRNGKey(42)
    jax_params = jax_model.init(rng, x_jax)["params"]
    
    # 4. Map PyTorch weights to JAX weights!
    # Convert PT weights dictionary to list of keys
    pt_dict = pt_model.state_dict()
    
    # Helper to print JAX param structure
    print("\n--- JAX Params Keys ---")
    def print_keys(d, prefix=""):
        for k, v in d.items():
            if isinstance(v, dict):
                print_keys(v, prefix + k + "/")
            else:
                print(f"{prefix}{k}: shape {v.shape}")
    print_keys(jax_params)
    
    # Helper to convert PT conv1d/linear weights to JAX dense weights
    # PT Conv1d weight shape: (out, in, 1) -> JAX Dense: (in, out)
    # PT Conv2d / Linear weight shape: (out, in) -> JAX Dense: (in, out)
    def to_jax_dense(pt_weight):
        w = pt_weight.detach().numpy()
        if w.ndim == 3: # Conv1d
            w = w[:, :, 0] # Remove kernel size dimension
        return np.transpose(w)
        
    def to_jax_bias(pt_bias):
        return pt_bias.detach().numpy()
        
    # Map weights
    new_params = jax.tree_util.tree_map(lambda x: x, jax_params)
    
    # Lifting (ChannelMLP_0)
    new_params['ChannelMLP_0']['dense_0']['kernel'] = jnp.array(to_jax_dense(pt_dict['lifting.fcs.0.weight']))
    new_params['ChannelMLP_0']['dense_0']['bias'] = jnp.array(to_jax_bias(pt_dict['lifting.fcs.0.bias']))
    new_params['ChannelMLP_0']['dense_1']['kernel'] = jnp.array(to_jax_dense(pt_dict['lifting.fcs.1.weight']))
    new_params['ChannelMLP_0']['dense_1']['bias'] = jnp.array(to_jax_bias(pt_dict['lifting.fcs.1.bias']))
    
    # Projection (ChannelMLP_1)
    new_params['ChannelMLP_1']['dense_0']['kernel'] = jnp.array(to_jax_dense(pt_dict['projection.fcs.0.weight']))
    new_params['ChannelMLP_1']['dense_0']['bias'] = jnp.array(to_jax_bias(pt_dict['projection.fcs.0.bias']))
    new_params['ChannelMLP_1']['dense_1']['kernel'] = jnp.array(to_jax_dense(pt_dict['projection.fcs.1.weight']))
    new_params['ChannelMLP_1']['dense_1']['bias'] = jnp.array(to_jax_bias(pt_dict['projection.fcs.1.bias']))
    
    # FNOBlocks
    for i in range(4):
        block_jax = f'FNOBlock_{i}'
        
        # Skip connection (SkipConnection_0)
        # Note: In PyTorch, skip_type="linear" uses Flattened1dConv which has a conv inside.
        new_params[block_jax]['SkipConnection_0']['Dense_0']['kernel'] = jnp.array(to_jax_dense(pt_dict[f'fno_blocks.fno_skips.{i}.conv.weight']))
        if pt_model.fno_blocks.fno_skips[i].conv.bias is not None:
            new_params[block_jax]['SkipConnection_0']['Dense_0']['bias'] = jnp.array(to_jax_bias(pt_dict[f'fno_blocks.fno_skips.{i}.conv.bias']))
        
        # Spectral Conv
        pt_w = pt_dict[f'fno_blocks.convs.{i}.weight.tensor'].detach().numpy()
        w1 = pt_w[:, :, :8, :]
        w2 = pt_w[:, :, -8:, :]
        new_params[block_jax]['SpectralConv_0']['weights_1'] = jnp.array(w1)
        new_params[block_jax]['SpectralConv_0']['weights_2'] = jnp.array(w2)
        if pt_model.fno_blocks.convs[i].bias is not None:
            new_params[block_jax]['SpectralConv_0']['bias'] = jnp.array(pt_dict[f'fno_blocks.convs.{i}.bias'].detach().numpy()[:, 0, 0])
            
        # ChannelMLP inside FNOBlock
        new_params[block_jax]['ChannelMLP_0']['dense_0']['kernel'] = jnp.array(to_jax_dense(pt_dict[f'fno_blocks.channel_mlp.{i}.fcs.0.weight']))
        new_params[block_jax]['ChannelMLP_0']['dense_0']['bias'] = jnp.array(to_jax_bias(pt_dict[f'fno_blocks.channel_mlp.{i}.fcs.0.bias']))
        new_params[block_jax]['ChannelMLP_0']['dense_1']['kernel'] = jnp.array(to_jax_dense(pt_dict[f'fno_blocks.channel_mlp.{i}.fcs.1.weight']))
        new_params[block_jax]['ChannelMLP_0']['dense_1']['bias'] = jnp.array(to_jax_bias(pt_dict[f'fno_blocks.channel_mlp.{i}.fcs.1.bias']))
        
        # SoftGating skip
        sg_w = pt_dict[f'fno_blocks.channel_mlp_skips.{i}.weight'].detach().numpy()
        new_params[block_jax]['SkipConnection_1']['SoftGating_0']['weight'] = jnp.array(np.transpose(sg_w, (0, 2, 3, 1)))

    print("\n--- Running Forward Passes ---")
    with torch.no_grad():
        out_pt = pt_model(x_pt).numpy()
        
    out_jax = jax_model.apply({"params": new_params}, x_jax)
    
    # Compare
    out_pt_transposed = np.transpose(out_pt, (0, 2, 3, 1))
    
    diff = np.abs(np.array(out_jax) - out_pt_transposed)
    print(f"Max difference: {np.max(diff):.6e}")
    print(f"Mean difference: {np.mean(diff):.6e}")
    print(f"PT mean: {np.mean(out_pt):.6e}, std: {np.std(out_pt):.6e}")
    print(f"JAX mean: {np.mean(out_jax):.6e}, std: {np.std(out_jax):.6e}")
    
    print("\n--- Map completed ---")
if __name__ == "__main__":
    main()
