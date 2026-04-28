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
from typing import Optional, Callable

class PhysicsAttention(nn.Module):
    """
    Universal Physics-aware Attention (Multi-Head Slice-based Attention) for Transolver.
    
    Acts as the standard 'Physics_Attention_Irregular_Mesh' from the original 
    Transolver repository. This version is suitable for irregular meshes in 
    1D, 2D, or 3D space and handles arbitrary point clouds.

    Key features:
    - Multi-head slicing for varied physical basis learning.
    - Learned temperature scaling with safety clamping.
    - Normalized slice aggregation for point-density independence.

    Attributes:
        dim (int): Dimensionality of the input and output features.
        heads (int): Number of attention heads.
        dim_head (int): Dimensionality per attention head.
        dropout (float): Dropout probability.
        slice_num (int): Number of physical slices (M).
    """
    dim: int
    heads: int = 8
    dim_head: int = 64
    dropout: float = 0.0
    slice_num: int = 64

    @nn.compact
    def __call__(self, x: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        B, N, C = x.shape
        inner_dim = self.heads * self.dim_head
        scale = self.dim_head ** -0.5
        
        # 1. Learned Temperature (B, H, 1, 1)
        # Clamped between 0.1 and 5.0 for stability as per structured mesh versions
        temperature = self.param('temperature', 
                                 lambda key, shape: jnp.ones(shape) * 0.5, 
                                 (1, self.heads, 1, 1))
        clamped_temp = jnp.clip(temperature, 0.1, 5.0)

        # 2. Input Projections (Point-wise Dense)
        # Separate features for aggregation (fx) and slice attribution (x)
        fx_mid_proj = nn.Dense(inner_dim, name="in_project_fx")(x)
        x_mid_proj = nn.Dense(inner_dim, name="in_project_x")(x)

        # Reshape and permute to (Batch, Heads, Points, Dim_Head)
        # Corresponds to Torch permute(0, 2, 1, 3)
        fx_mid = fx_mid_proj.reshape(B, N, self.heads, self.dim_head).transpose(0, 2, 1, 3)
        x_mid = x_mid_proj.reshape(B, N, self.heads, self.dim_head).transpose(0, 2, 1, 3)

        # 3. Slicing Mechanism
        # Orthogonal initialization is used for principled basis function learning
        slice_proj = nn.Dense(
            self.slice_num, 
            use_bias=True, 
            kernel_init=nn.initializers.xavier_uniform(), 
            name="in_project_slice"
        )
        
        # Softmax over slice dimension (G) -> (B, H, N, G)
        slice_weights = nn.softmax(slice_proj(x_mid) / clamped_temp, axis=-1)
        
        # Sum weights over points (N) to get normalization factor (B, H, 1, G)
        slice_norm = jnp.sum(slice_weights, axis=2, keepdims=True)
        
        # Aggregate: slice_token = slice_weights^T @ fx_mid -> (B, H, G, D_H)
        slice_token = jnp.einsum("bhnc,bhng->bhgc", fx_mid, slice_weights)
        
        # Normalize to account for varying mesh point densities
        slice_token = slice_token / (slice_norm.transpose(0, 1, 3, 2) + 1e-5)

        # 4. Attention in Slice Space (Multi-Head Interaction)
        q = nn.Dense(self.dim_head, use_bias=False, name="to_q")(slice_token)
        k = nn.Dense(self.dim_head, use_bias=False, name="to_k")(slice_token)
        v = nn.Dense(self.dim_head, use_bias=False, name="to_v")(slice_token)

        # Slice-wise correlation matrix: (B, H, G, G)
        dots = jnp.matmul(q, k.transpose(0, 1, 3, 2)) * scale
        attn = nn.softmax(dots, axis=-1)
        
        if not deterministic and self.dropout > 0:
            attn = nn.Dropout(self.dropout)(attn, deterministic=False)
            
        # Refined physical states (B, H, G, D_H)
        out_slice_token = jnp.matmul(attn, v)

        # 5. Deslicing (Broadcasting)
        # out_x = out_slice_token @ slice_weights^T -> (B, H, N, D_H)
        out_x = jnp.einsum("bhgc,bhng->bhnc", out_slice_token, slice_weights)
        
        # Return to (Batch, Points, Channels)
        out_x = out_x.transpose(0, 2, 1, 3).reshape(B, N, inner_dim)
        
        # Final linear projection and dropout (O(NC) operation)
        out = nn.Dense(self.dim, name="to_out_0")(out_x)
        if not deterministic and self.dropout > 0:
            out = nn.Dropout(self.dropout, name="to_out_1")(out, deterministic=False)
            
        return out
