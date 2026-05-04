import jax.numpy as jnp
from flax import linen as nn
from typing import Tuple, Callable

from .channel_mlp import ChannelMLP
from .hankel_conv import HankelConv

class HNOBlock(nn.Module):
    """
    A single Hankel Neural Operator Block.
    """
    hidden_channels: int
    n_modes: Tuple[int, int]
    activation: Callable = nn.gelu
    use_norm: bool = False
    skip_type: str = "linear"
    channel_mlp_skip: str = "soft-gating"
    use_channel_mlp: bool = True
    order: int = 0
    r_axis: int = 1
    theta_axis: int = 2

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x_ht = HankelConv(
            in_channels=self.hidden_channels,
            out_channels=self.hidden_channels,
            n_modes=self.n_modes,
            order=self.order,
            r_axis=self.r_axis,
            theta_axis=self.theta_axis
        )(x)
        
        if self.skip_type == "linear":
            skip = nn.Dense(self.hidden_channels, use_bias=False)(x)
        elif self.skip_type == "identity":
            skip = x
        else:
            skip = 0
            
        x_out = x_ht + skip
        if self.use_norm:
            x_out = nn.GroupNorm(num_groups=1)(x_out)
            
        x_out = self.activation(x_out)
        
        if self.use_channel_mlp:
            mlp_out = ChannelMLP(
                out_channels=self.hidden_channels,
                hidden_channels=self.hidden_channels * 2,
                n_layers=2,
                activation=self.activation
            )(x_out)
            
            if self.channel_mlp_skip == "soft-gating":
                gate = nn.Dense(self.hidden_channels)(x_out)
                x_out = mlp_out * nn.sigmoid(gate) + x_out
            elif self.channel_mlp_skip == "linear":
                skip = nn.Dense(self.hidden_channels, use_bias=False)(x_out)
                x_out = mlp_out + skip
            else:
                x_out = mlp_out + x_out
                
        return x_out
