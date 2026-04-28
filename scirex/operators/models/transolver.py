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

from typing import Callable, Optional
from flax import linen as nn
import jax.numpy as jnp
from ..layers.transolver_block import TransolverBlock
from ..layers.transolver_mlp import TransolverMLP

class Transolver(nn.Module):
    """
    Transolver: A Transformer-based framework for learning solution operators of PDEs.
    
    This model implements the top-level architecture of Transolver, which is 
    mesh-agnostic and achieves linear complexity O(NM) where N is the number 
    of points and M is the number of physical slices.
    
    The architecture follows a lifting-processing-projection pipeline:
    [Input] -> Lifting (TransolverMLP) -> Stacked TransolverBlocks -> Projection (TransolverMLP) -> [Output]

    Attributes:
        hidden_channels (int): Dimensionality of the latent representation.
        out_channels (int): Dimensionality of the output target space.
        n_layers (int): Number of iterative Transolver blocks.
        n_heads (int): Number of attention heads in the physics-attention layer.
        dim_head (int): Dimension per head.
        slice_num (int): Number of learnable physical slices (M).
        mlp_ratio (float): Expansion factor for the hidden layers in the MLP.
        dropout (float): Dropout probability.
        activation (Callable): Activation function (default: nn.gelu).
    """
    hidden_channels: int
    out_channels: int
    n_layers: int
    n_heads: int = 8
    dim_head: int = 64
    slice_num: int = 64
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    activation: Callable = nn.gelu

    @nn.compact
    def __call__(self, x: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        """
        Forward pass of the Transolver model.
        
        Args:
            x (jnp.ndarray): Input features/coordinates of shape (batch, num_points, in_channels).
            deterministic (bool): Whether to use dropout (set to True during inference).
            
        Returns:
            jnp.ndarray: Predicted fields of shape (batch, num_points, out_channels).
        """
        
        # 1. Lifting Stage (Encoder)
        # Maps input points to the latent hidden space
        x = TransolverMLP(
            n_hidden=self.hidden_channels,
            n_output=self.hidden_channels,
            n_layers=1,
            activation=self.activation,
            residual=False,
            name="lifting"
        )(x)
        
        # 2. Backbone Processing Stage (Transformer blocks)
        # Global interaction and local refinement across the domain
        for i in range(self.n_layers):
            x = TransolverBlock(
                dim=self.hidden_channels,
                heads=self.n_heads,
                dim_head=self.dim_head,
                dropout=self.dropout,
                slice_num=self.slice_num,
                mlp_hidden=int(self.hidden_channels * self.mlp_ratio),
                name=f"block_{i}"
            )(x, deterministic=deterministic)
            
        # 3. Projection Stage (Decoder)
        # Maps latent representation to the physical output space
        x = TransolverMLP(
            n_hidden=self.hidden_channels,
            n_output=self.out_channels,
            n_layers=1,
            activation=self.activation,
            residual=False,
            name="projection"
        )(x)
        
        return x
