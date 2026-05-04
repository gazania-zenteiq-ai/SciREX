import jax.numpy as jnp
from flax import linen as nn
from typing import Tuple, Literal, Union, List, Callable

from ..layers.channel_mlp import ChannelMLP
from ..layers.embeddings import GridEmbedding
from ..layers.padding import DomainPadding
from ..layers.hno_block import HNOBlock

class HNO(nn.Module):
    """
    2D Hankel Neural Operator (HNO).
    
    This architecture mimics the Fourier Neural Operator (FNO) but replaces the 
    Fast Fourier Transform (FFT) with the Fast Hankel Transform (FHT) across the 
    radial dimension, while retaining RFFT across the angular (theta) dimension.
    """
    hidden_channels: int
    n_layers: int
    n_modes: Tuple[int, int]
    out_channels: int
    lifting_channel_ratio: int = 2
    projection_channel_ratio: int = 2
    use_grid: bool = True
    hno_skip: Literal["identity", "linear", "soft-gating"] = "linear"
    channel_mlp_skip: Literal["identity", "linear", "soft-gating"] = "soft-gating"
    use_channel_mlp: bool = True
    padding: Union[float, List[float]] = 0.0
    use_norm: bool = False
    activation: Callable = nn.gelu
    order: int = 0  # Order of the Bessel Function
    r_axis: int = 1
    theta_axis: int = 2

    @nn.compact
    def __call__(self, x: jnp.ndarray, pde_loss: bool = True) -> jnp.ndarray:
        n_dim = 2
        original_shape = x.shape
        
        needs_pad = (
            any(p > 0 for p in self.padding)
            if isinstance(self.padding, (list, tuple))
            else self.padding > 0
        )
        
        if self.use_grid:
            grid_boundaries = tuple((0.0, 1.0) for _ in range(n_dim))
            x = GridEmbedding(grid_boundaries=grid_boundaries)(x)
            
        if needs_pad:
            pad_layer = DomainPadding(padding=self.padding)
            x = pad_layer(x)
            
        lifting_hidden = self.hidden_channels * self.lifting_channel_ratio
        x = ChannelMLP(
            out_channels=self.hidden_channels,
            hidden_channels=lifting_hidden,
            n_layers=2,
            activation=self.activation,
            name="lifting_layer"
        )(x)
        
        for i in range(self.n_layers):
            x = HNOBlock(
                hidden_channels=self.hidden_channels,
                n_modes=self.n_modes,
                activation=self.activation,
                use_norm=self.use_norm,
                skip_type=self.hno_skip,
                channel_mlp_skip=self.channel_mlp_skip,
                use_channel_mlp=self.use_channel_mlp,
                order=self.order,
                r_axis=self.r_axis,
                theta_axis=self.theta_axis
            )(x)
            if i == self.n_layers - 1:
                self.sow("intermediates", "last", x)
                
        projection_hidden = self.hidden_channels * self.projection_channel_ratio
        x = ChannelMLP(
            out_channels=self.out_channels,
            hidden_channels=projection_hidden,
            n_layers=2,
            activation=self.activation,
            name="projection_layer"
        )(x)
        
        if needs_pad:
            x = pad_layer(x, inverse=True, original_shape=original_shape)
            
        return x
