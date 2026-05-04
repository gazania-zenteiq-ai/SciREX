import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Tuple, Optional
from .fast_hankel import fht, ifht

class HankelConv(nn.Module):
    """
    2D Hankel-Fourier Convolution layer.
    
    Transforms the radial dimension using Fast Hankel Transform (FHT) and 
    the angular (theta) dimension using Real Fast Fourier Transform (RFFT).
    """
    in_channels: int
    out_channels: int
    n_modes: Tuple[int, int]
    init_std: Optional[float] = None
    order: int = 0
    r_axis: int = 1
    theta_axis: int = 2

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # Expected x shape: (batch, dim_r, dim_theta, in_channels) or similar
        batch = x.shape[0]
        dim_r = x.shape[self.r_axis]
        dim_theta = x.shape[self.theta_axis]
        
        # 1. Forward Transforms
        # Radial transform (FHT yields real values)
        x_h = fht(x, order=self.order, axes=(self.r_axis,))
        # Angular transform (RFFT yields complex values)
        x_ht = jnp.fft.rfft(x_h, axis=self.theta_axis, norm="ortho")
        
        # 2. Weights Initialization
        scale = 0.05 if self.init_std is None else self.init_std
        modes_r, modes_theta = self.n_modes
        
        # Weights shape: (in_channels, out_channels, modes_r, modes_theta)
        weights_shape = (self.in_channels, self.out_channels, modes_r, modes_theta)
        w = self.param('weights', jax.nn.initializers.normal(stddev=scale), weights_shape, jnp.complex64)
        
        # 3. Create output tensor in frequency domain
        out_ht_shape = list(x.shape)
        out_ht_shape[self.theta_axis] = dim_theta // 2 + 1
        out_ht_shape[-1] = self.out_channels
        out_ht = jnp.zeros(out_ht_shape, dtype=jnp.complex64)
        
        # 4. Mode Filtering
        slices_in = [slice(None)] * x_ht.ndim
        slices_out = [slice(None)] * len(out_ht_shape)
        
        slices_in[self.r_axis] = slice(None, modes_r)
        slices_in[self.theta_axis] = slice(None, modes_theta)
        
        slices_out[self.r_axis] = slice(None, modes_r)
        slices_out[self.theta_axis] = slice(None, modes_theta)
        
        x_corner = x_ht[tuple(slices_in)]
        
        # Dynamic einsum based on axis positions
        # e.g., if r_axis=1, theta_axis=2: bxyi,ioxy->bxyo
        letters = ['b', '', '', 'i']
        letters[self.r_axis] = 'x'
        letters[self.theta_axis] = 'y'
        in_str = "".join(letters)
        
        out_letters = letters.copy()
        out_letters[-1] = 'o'
        out_str = "".join(out_letters)
        
        einsum_str = f"{in_str},ioxy->{out_str}"
        
        out_corner = jnp.einsum(einsum_str, x_corner, w)
        out_ht = out_ht.at[tuple(slices_out)].set(out_corner)
        
        # 5. Inverse Transforms
        x_out = jnp.fft.irfft(out_ht, n=dim_theta, axis=self.theta_axis, norm="ortho")
        x_out = ifht(x_out, order=self.order, axes=(self.r_axis,))
        
        return x_out
