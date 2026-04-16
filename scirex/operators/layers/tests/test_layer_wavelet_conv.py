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
Unit tests for WaveletConv.

Tests are written in N-D style to ensure the block works
for arbitrary spatial dimensionalities.
"""

import jax
import jax.numpy as jnp
import pytest

from scirex.operators.layers.wavelet_conv import WaveletConv

@pytest.mark.parametrize(
    "spatial_shape,size",
    [
        ((64,), 64),                 # 1D
        ((32, 32), (32, 32)),         # 2D
        ((16, 16, 16), (16, 16, 16)), # 3D
    ],
)
def test_wavelet_conv_forward_nd(spatial_shape, size):
    """WaveletConv should process shape and map hidden channels dynamically."""
    rng = jax.random.PRNGKey(0)
    batch = 2
    in_channels = 3
    out_channels = 5

    x = jnp.ones((batch, *spatial_shape, in_channels))

    model = WaveletConv(
        in_channels=in_channels,
        out_channels=out_channels,
        level=2,
        size=size,
        wavelet="db4",
        mode="symmetric"
    )

    params = model.init(rng, x)
    y = model.apply(params, x)

    # Spatial shape should be preserved, output dimension mapped to out_channels
    assert y.shape == (batch, *spatial_shape, out_channels)

def test_wavelet_conv_padding():
    """WaveletConv should safely pad any generic shapes that aren't strict powers of 2."""
    rng = jax.random.PRNGKey(0)
    batch = 2
    in_channels = 4
    out_channels = 8

    # 30 is not a standard clean power of two scaling boundary natively 
    x = jax.random.normal(rng, (batch, 30, 30, in_channels))

    model = WaveletConv(
        in_channels=in_channels,
        out_channels=out_channels,
        level=2,
        size=(30, 30),
        wavelet="db2",
        mode="symmetric"
    )

    params = model.init(rng, x)
    y = model.apply(params, x)

    # Even with automated padding inside, it should strictly crop cleanly back outwards
    assert y.shape == (batch, 30, 30, out_channels)
