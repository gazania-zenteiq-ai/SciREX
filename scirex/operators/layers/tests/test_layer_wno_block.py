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
Unit tests for WaveletBlock.

Tests are written in N-D style to ensure the block works
for arbitrary spatial dimensionalities.
"""

import jax
import jax.numpy as jnp
import pytest

from scirex.operators.layers.wno_block import WNOBlock

@pytest.mark.parametrize(
    "spatial_shape,size",
    [
        ((64,), 64),                 # 1D
        ((32, 32), (32, 32)),         # 2D
        ((16, 16, 16), (16, 16, 16)), # 3D
    ],
)
def test_wno_block_forward_nd(spatial_shape, size):
    """WaveletBlock should map hidden channels and preserve input shape dimensionality."""
    rng = jax.random.PRNGKey(0)
    batch = 2
    in_channels = 3
    hidden_channels = 5

    x = jax.random.normal(rng, (batch, *spatial_shape, in_channels))

    model = WNOBlock(
        hidden_channels=hidden_channels,
        level=2,
        size=size,
        wavelet="db4"
    )

    params = model.init(rng, x)
    y = model.apply(params, x)

    assert y.shape == (batch, *spatial_shape, hidden_channels)


@pytest.mark.parametrize(
    "skip_type",
    ["identity", "linear", "soft-gating"],
)
def test_wno_block_skip_types(skip_type):
    """WaveletBlock should successfully route parameters alongside skip types."""
    rng = jax.random.PRNGKey(0)
    batch = 2
    spatial_shape = (16, 16)
    channels = 10

    x = jnp.ones((batch, *spatial_shape, channels))

    model = WNOBlock(
        hidden_channels=channels,
        level=1,
        size=spatial_shape,
        wavelet="db2",
        skip_type=skip_type,
    )

    params = model.init(rng, x)
    y = model.apply(params, x)

    assert y.shape == x.shape
