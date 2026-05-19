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
from flax import linen as nn
from typing import Tuple, Optional
import itertools


class SpectralConv(nn.Module):
    """N-dimensional Spectral Convolution layer (supports 1D, 2D, 3D, and beyond)."""

    in_channels: int
    out_channels: int
    n_modes: Tuple[int, ...]
    init_std: Optional[float] = None

    bias: bool = True

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # x shape: (batch, dim1, dim2, ..., dimN, in_channels)
        n_dim = len(self.n_modes)
        batch = x.shape[0]
        spatial_dims = x.shape[1:-1]

        # 0. Safety Check: Ensure requested modes don't exceed Nyquist limits
        for i, (mode, dim) in enumerate(zip(self.n_modes, spatial_dims)):
            # The last dimension is roughly halved due to RFFT
            max_modes = dim // 2 + 1 if i == n_dim - 1 else dim
            if mode > max_modes:
                raise ValueError(
                    f"n_modes[{i}]={mode} exceeds maximum allowed modes ({max_modes}) "
                    f"for spatial dimension size {dim}."
                )

        # 1. FFT
        axes = tuple(range(1, n_dim + 1))
        x_ft = jnp.fft.rfftn(x, axes=axes, norm="backward")

        # 2. Weights Initialization
        scale = (2.0 / (self.in_channels + self.out_channels)) ** 0.5 if self.init_std is None else self.init_std
        half_modes = tuple(m // 2 for m in self.n_modes[:-1]) + (self.n_modes[-1] // 2 + 1,)
        weights_shape = (self.in_channels, self.out_channels) + half_modes

        # For N dimensions, the number of corners in frequency space is 2**(N-1).
        n_corners = 2 ** (n_dim - 1)
        
        # PyTorch Adam treats real and imaginary parts of complex parameters as independent real parameters.
        # optax.adamw treats a complex parameter as a single parameter and uses |g|^2 for variance, coupling their learning rates.
        # To achieve parity, we split the complex weights into explicit real and imaginary parameters!
        weights_r = [
            self.param(
                f"weights_r_{i+1}",
                jax.nn.initializers.normal(stddev=scale / (2**0.5)),
                weights_shape,
                jnp.float32,
            )
            for i in range(n_corners)
        ]
        weights_i = [
            self.param(
                f"weights_i_{i+1}",
                jax.nn.initializers.normal(stddev=scale / (2**0.5)),
                weights_shape,
                jnp.float32,
            )
            for i in range(n_corners)
        ]

        # Create output tensor in frequency domain
        out_ft_shape = (
            (batch,)
            + spatial_dims[:-1]
            + (spatial_dims[-1] // 2 + 1,)
            + (self.out_channels,)
        )
        out_ft = jnp.zeros(out_ft_shape, dtype=jnp.complex64)

        # 3. Build dynamic einsum string (safely avoids hardcoded letter limits)
        # We reserve 'b' for batch, 'i' for in_channels, 'o' for out_channels
        available_letters = "acdefghjklmnpqrstuvwxyz"
        if n_dim > len(available_letters):
            raise ValueError(
                f"Too many spatial dimensions ({n_dim}) for einsum string generation."
            )
        spatial_letters = available_letters[:n_dim]
        einsum_str = f"b{spatial_letters}i,io{spatial_letters}->b{spatial_letters}o"

        # 4. Apply Weights to Frequency Corners
        corner_idx = 0
        for signs in itertools.product([1, -1], repeat=n_dim - 1):
            slices_in = [slice(None)]  # batch
            slices_out = [slice(None)]  # batch

            for d, sign in enumerate(signs):
                modes = half_modes[d]
                if sign == 1:
                    slices_in.append(slice(None, modes))
                    slices_out.append(slice(None, modes))
                else:
                    slices_in.append(slice(-modes, None))
                    slices_out.append(slice(-modes, None))

            # Last spatial dimension (always positive frequencies for RFFT)
            last_modes = half_modes[-1]
            slices_in.append(slice(None, last_modes))
            slices_out.append(slice(None, last_modes))

            # Channel dimension
            slices_in.append(slice(None))
            slices_out.append(slice(None))

            # Extract corner, multiply, and inject back
            x_corner = x_ft[tuple(slices_in)]
            wr = weights_r[corner_idx]
            wi = weights_i[corner_idx]

            # Bypass complex einsum unsupported functionality issue on CPU
            xr, xi = jnp.real(x_corner), jnp.imag(x_corner)
            out_r = jnp.einsum(einsum_str, xr, wr) - jnp.einsum(einsum_str, xi, wi)
            out_i = jnp.einsum(einsum_str, xr, wi) + jnp.einsum(einsum_str, xi, wr)
            out_corner = out_r + 1j * out_i

            out_ft = out_ft.at[tuple(slices_out)].set(out_corner)

            corner_idx += 1

        # 5. Inverse FFT
        x = jnp.fft.irfftn(out_ft, s=spatial_dims, axes=axes, norm="backward")

        # 6. Learnable Bias
        if self.bias:
            bias_val = self.param(
                "bias",
                jax.nn.initializers.normal(stddev=scale),
                (self.out_channels,),
            )
            # Reshape bias for broadcasting channels-last: (1, ..., 1, out_channels)
            reshape_shape = [1] * (n_dim + 2)
            reshape_shape[-1] = self.out_channels
            x = x + bias_val.reshape(reshape_shape)

        return x
