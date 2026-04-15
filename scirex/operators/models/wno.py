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

from typing import Callable, List, Literal, Optional, Tuple, Union

import jax.numpy as jnp
from flax import linen as nn

from ..layers.channel_mlp import ChannelMLP
from ..layers.embeddings import GridEmbedding
from ..layers.padding import DomainPadding
from ..layers.wavelet_block import WaveletBlock, mish


class WNO(nn.Module):
    """
    N-dimensional Wavelet Neural Operator.

    The spatial dimensionality is inferred from the input tensor shape
    ``(batch, spatial_1, ..., spatial_n, channels)``. This keeps the model
    usable across 1D, 2D, and 3D operator-learning problems without
    maintaining separate model implementations.
    """

    hidden_channels: int
    n_layers: int
    level: int
    size: Union[int, Tuple[int, ...]]
    out_channels: int
    wavelet: str = "db4"
    mode: str = "symmetric"
    lifting_channel_ratio: int = 2
    projection_channel_ratio: int = 2
    use_grid: bool = True
    grid_boundaries: Optional[Tuple[Tuple[float, float], ...]] = None
    padding: Union[float, List[float]] = 0.0
    skip_type: Literal["identity", "linear", "soft-gating"] = "linear"
    activation: Callable = mish

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        ndim = x.ndim - 2
        if ndim < 1:
            raise ValueError("WNO expects at least one spatial dimension.")

        original_shape = x.shape
        needs_pad = (
            any(p > 0 for p in self.padding)
            if isinstance(self.padding, (list, tuple))
            else self.padding > 0
        )

        if self.use_grid:
            if self.grid_boundaries is None:
                grid_boundaries = tuple((0.0, 1.0) for _ in range(ndim))
            else:
                grid_boundaries = self.grid_boundaries
                if len(grid_boundaries) != ndim:
                    raise ValueError(
                        f"grid_boundaries has {len(grid_boundaries)} entries, expected {ndim}."
                    )
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
        )(x)

        for layer_idx in range(self.n_layers):
            block_activation = self.activation if layer_idx < self.n_layers - 1 else None
            x = WaveletBlock(
                hidden_channels=self.hidden_channels,
                level=self.level,
                size=self.size,
                wavelet=self.wavelet,
                mode=self.mode,
                activation=block_activation,
                skip_type=self.skip_type,
            )(x)

        projection_hidden = self.hidden_channels * self.projection_channel_ratio
        x = ChannelMLP(
            out_channels=self.out_channels,
            hidden_channels=projection_hidden,
            n_layers=2,
            activation=self.activation,
        )(x)

        if needs_pad:
            x = pad_layer(x, inverse=True, original_shape=original_shape)

        return x
