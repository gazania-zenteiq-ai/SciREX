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

import jax
import jax.numpy as jnp
from typing import Tuple, Optional
import itertools
from flax import linen as nn

# class SpectralConvDerv(nn.Module):
#     """
#     N-dimensional Spectral Convolution layer (supports 1D, 2D, 3D, and beyond).
    
#     The dimensionality is automatically inferred from the length of `n_modes`.
    
#     This layer performs a convolution in the Fourier domain by:
#     1. Transforming the input to the frequency domain using a Real FFT (RFFT).
#     2. Multiplying the lower Fourier modes by learnable complex weights.
#     3. Inverse transforming the filtered signal back to the spatial domain.
    
#     Attributes:
#         in_channels (int): Number of input channels.
#         out_channels (int): Number of output channels.
#         n_modes (Tuple[int, ...]): Number of Fourier modes to retain for each spatial dimension.
#         init_std (float, optional): Standard deviation for weight initialization.
#     """         
#     in_channels: int
#     out_channels: int
#     n_modes: Tuple[int, ...]
#     init_std: Optional[float] = None
#     deriv_order: int              # 1 for first derivative, 2 for second

#     @nn.compact
#     def __call__(self, x: jnp.ndarray, derv_order:int) -> jnp.ndarray:
#         # x shape: (batch, dim1, dim2, ..., dimN, in_channels)
#         n_dim = len(self.n_modes)
#         batch = x.shape[0]
#         spatial_dims = x.shape[1:-1]
        
#         # 0. Safety Check: Ensure requested modes don't exceed Nyquist limits
#         for i, (mode, dim) in enumerate(zip(self.n_modes, spatial_dims)):
#             # The last dimension is roughly halved due to RFFT
#             max_modes = dim // 2 + 1 if i == n_dim - 1 else dim
#             if mode > max_modes:
#                 raise ValueError(
#                     f"n_modes[{i}]={mode} exceeds maximum allowed modes ({max_modes}) "
#                     f"for spatial dimension size {dim}."
#                 )

#         # 1. FFT
#         axes = tuple(range(1, n_dim + 1))
#         x_ft = jnp.fft.rfftn(x, axes=axes, norm="ortho")
        
#         # 2. Weights Initialization
#         scale = 0.05 if self.init_std is None else self.init_std
#         weights_shape = (self.in_channels, self.out_channels) + self.n_modes
        
#         # For N dimensions, the number of corners in frequency space is 2**(N-1).
#         n_corners = 2**(n_dim - 1)
#         weights = [
#             self.param(f'weights_{i+1}', jax.nn.initializers.normal(stddev=scale), weights_shape, jnp.complex64)
#             for i in range(n_corners)
#         ]
        
#         # Create output tensor in frequency domain
#         out_ft_shape = (batch,) + spatial_dims[:-1] + (spatial_dims[-1] // 2 + 1,) + (self.out_channels,)
#         out_ft = jnp.zeros(out_ft_shape, dtype=jnp.complex64)
        
#         # 3. Build dynamic einsum string (safely avoids hardcoded letter limits)
#         # We reserve 'b' for batch, 'i' for in_channels, 'o' for out_channels
#         available_letters = "acdefghjklmnpqrstuvwxyz"
#         if n_dim > len(available_letters):
#              raise ValueError(f"Too many spatial dimensions ({n_dim}) for einsum string generation.")
#         spatial_letters = available_letters[:n_dim]
#         einsum_str = f"b{spatial_letters}i,io{spatial_letters}->b{spatial_letters}o"
        
#         # 4. Apply Weights to Frequency Corners
#         corner_idx = 0
#         for signs in itertools.product([1, -1], repeat=n_dim - 1):
#             slices_in = [slice(None)]  # batch
#             slices_out = [slice(None)] # batch
            
#             for d, sign in enumerate(signs):
#                 modes = self.n_modes[d]
#                 if sign == 1:
#                     slices_in.append(slice(None, modes))
#                     slices_out.append(slice(None, modes))
#                 else:
#                     slices_in.append(slice(-modes, None))
#                     slices_out.append(slice(-modes, None))
                    
#             # Last spatial dimension (always positive frequencies for RFFT)
#             last_modes = self.n_modes[-1]
#             slices_in.append(slice(None, last_modes))
#             slices_out.append(slice(None, last_modes))
            
#             # Channel dimension
#             slices_in.append(slice(None))
#             slices_out.append(slice(None))
            
#             # Extract corner, multiply, and inject back
#             x_corner = x_ft[tuple(slices_in)]
#             w_corner = weights[corner_idx]
            
#             out_corner = jnp.einsum(einsum_str, x_corner, w_corner)
#             out_ft = out_ft.at[tuple(slices_out)].set(out_corner)
            
#             corner_idx += 1
            
#         # ---------------------------------------------------------------------
#         # 5 Compute the Spatial Derivative (i * k multiplier)
#         # ---------------------------------------------------------------------
#         nx,ny = x.shape[1], x.shape[2] 
#         Lx,Ly = 1,1 # Assuming 2D spatial dimensions for derivative computation
#         k_x = jnp.fft.fftfreq(nx) * nx
#         k_broadcast_x = ((2.0 * jnp.pi / Lx) * k_x).reshape((1, nx, 1, 1))
#         mult_x = (1j * k_broadcast_x) ** self.deriv_order
        
#         # d/dy multiplier
#         k_y = jnp.fft.rfftfreq(ny) * ny
#         k_broadcast_y = ((2.0 * jnp.pi / Ly) * k_y).reshape((1, 1, ny // 2 + 1, 1))
#         mult_y = (1j * k_broadcast_y) ** self.deriv_order
        
#         # Branch the Fourier tensor
#         out_ft_x = out_ft * mult_x
#         out_ft_y = out_ft * mult_y
        
#         # ---------------------------------------------------------------------
#         # 5. Inverse 2D FFT on both branches
#         # ---------------------------------------------------------------------
#         dv_dx = jnp.fft.irfftn(out_ft_x, s=(nx, ny), axes=(1, 2), norm="ortho")
#         dv_dy = jnp.fft.irfftn(out_ft_y, s=(nx, ny), axes=(1, 2), norm="ortho")
        
#         # Stack along a new final axis
#         # Resulting shape: (batch_size, nx, ny, channels, 2)
#         gradient_tensor = jnp.stack([dv_dx, dv_dy], axis=-1)
        
#         return gradient_tensor


def exact_spectral_derivative(x: jnp.ndarray, deriv_order: int) -> jnp.ndarray:
    """
    Computes the exact spatial derivative in the Fourier domain without learnable parameters.
    x shape: (batch, nx, ny, channels)
    """
    nx, ny = x.shape[1], x.shape[2]
    Lx, Ly = 1.0, 1.0  # Domain lengths
    
    # 1. Forward FFT
    x_ft = jnp.fft.rfftn(x, axes=(1, 2), norm="ortho")
    
    # 2. X-derivative multiplier
    k_x = jnp.fft.fftfreq(nx) * nx
    k_broadcast_x = ((2.0 * jnp.pi / Lx) * k_x).reshape((1, nx, 1, 1))
    mult_x = (1j * k_broadcast_x) ** deriv_order
    
    # 3. Y-derivative multiplier
    k_y = jnp.fft.rfftfreq(ny) * ny
    k_broadcast_y = ((2.0 * jnp.pi / Ly) * k_y).reshape((1, 1, ny // 2 + 1, 1))
    mult_y = (1j * k_broadcast_y) ** deriv_order
    
    # 4. Apply multipliers
    out_ft_x = x_ft * mult_x
    out_ft_y = x_ft * mult_y
    
    # 5. Inverse FFT
    dv_dx = jnp.fft.irfftn(out_ft_x, s=(nx, ny), axes=(1, 2), norm="ortho")
    dv_dy = jnp.fft.irfftn(out_ft_y, s=(nx, ny), axes=(1, 2), norm="ortho")
    
    # Stack resulting shape: (batch_size, nx, ny, channels, 2)
    return jnp.stack([dv_dx, dv_dy], axis=-1)