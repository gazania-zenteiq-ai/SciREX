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

import jax.numpy as jnp
from flax import linen as nn
from typing import Optional
from .physics_attention import PhysicsAttention
from .transolver_mlp import TransolverMLP

class TransolverBlock(nn.Module):
    """
    Standard block for the Transolver architecture.
    
    Following the transformer backbone pattern, each block consists of 
    a Physics-aware Attention sub-layer and a Feed-Forward (MLP) sub-layer, 
    both utilizing Pre-Layer Normalization and Residual Connections.
    
    Attributes:
        dim (int): Embedding dimension.
        heads (int): Number of attention heads.
        dim_head (int): Dimension per attention head.
        dropout (float): Dropout probability.
        slice_num (int): Number of physical slices (M).
        mlp_hidden (Optional[int]): Hidden dimension for the MLP. Defaults to 4 * dim.
        mlp_layers (int): Number of internal hidden layers in the MLP.
    """
    dim: int
    heads: int = 8
    dim_head: int = 64
    dropout: float = 0.0
    slice_num: int = 64
    mlp_hidden: Optional[int] = None
    mlp_layers: int = 1

    @nn.compact
    def __call__(self, x: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        # --- Sub-layer 1: Physics Attention ---
        # Implementation of x = x + Attention(LayerNorm(x))
        identity = x
        x = nn.LayerNorm(name="norm1")(x)
        x = PhysicsAttention(
            dim=self.dim,
            heads=self.heads,
            dim_head=self.dim_head,
            dropout=self.dropout,
            slice_num=self.slice_num,
            name="attn"
        )(x, deterministic=deterministic)
        x = identity + x

        # --- Sub-layer 2: Feed-Forward (MLP) ---
        # Implementation of x = x + MLP(LayerNorm(x))
        identity = x
        x = nn.LayerNorm(name="norm2")(x)
        
        # Default hidden dimension is usually 4x the model dimension
        n_hidden = self.mlp_hidden if self.mlp_hidden is not None else 4 * self.dim
        x = TransolverMLP(
            n_hidden=n_hidden,
            n_output=self.dim,
            n_layers=self.mlp_layers,
            residual=True,
            name="mlp"
        )(x)
        x = identity + x

        return x
