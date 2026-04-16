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

from typing import Callable, Literal, Tuple, Union

import jax.numpy as jnp
from flax import linen as nn

from .skip_connection import SkipConnection
from .wavelet_conv import WaveletConv


def mish(x: jnp.ndarray) -> jnp.ndarray:
    """
    Mish activation function: x * tanh(softplus(x)).

    Often provides smoother gradients compared to ReLU, making it suitable for
    operator learning tasks like WNO.
    """
    return x * jnp.tanh(nn.softplus(x))


class WNOBlock(nn.Module):
    """
    N-dimensional Wavelet Neural Operator (WNO) block.

    This block implements one WNO update step:
        v(j+1)(x) = sigma(K(v(j)) + W(v(j)))

    The spatial dimensionality is inferred from the input tensor shape
    ``(batch, spatial_1, ..., spatial_n, channels)``.
    """

    hidden_channels: int
    level: int = 1
    size: Union[int, Tuple[int, ...]] = 1024
    wavelet: str = "db4"
    mode: str = "symmetric"
    activation: Callable = mish
    skip_type: Literal["identity", "linear", "soft-gating"] = "linear"

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Apply the wavelet operator block to the input tensor.

        Args:
            x: Input tensor of shape ``(batch, ..., channels)``.

        Returns:
            Tensor of shape ``(batch, ..., hidden_channels)``.
        """
        y = WaveletConv(
            in_channels=x.shape[-1],
            out_channels=self.hidden_channels,
            level=self.level,
            size=self.size,
            wavelet=self.wavelet,
            mode=self.mode,
        )(x)

        shortcut = SkipConnection(
            out_channels=self.hidden_channels,
            skip_type=self.skip_type,
        )(x)

        out = y + shortcut
        if self.activation is not None:
            out = self.activation(out)

        return out
