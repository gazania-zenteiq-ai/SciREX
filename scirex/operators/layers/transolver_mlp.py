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
from typing import Callable

class TransolverMLP(nn.Module):
    """
    Multi-Layer Perceptron (MLP) block used in Transolver.
    
    This implementation follows the original Transolver's 'Sandwich' structure:
    1. A 'pre' linear layer that projects input to hidden dimension.
    2. 'N' hidden layers with optional internal residual connections.
    3. A 'post' linear layer that projects to the final output dimension.
    
    Attributes:
        n_hidden (int): Internal dimensionality of the MLP.
        n_output (int): Output dimensionality.
        n_layers (int): Number of internal hidden layers.
        activation (Callable): Activation function (default: nn.gelu).
        residual (bool): Whether to use residual connections in hidden layers.
    """
    n_hidden: int
    n_output: int
    n_layers: int = 1
    activation: Callable = nn.gelu
    residual: bool = True

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # Step 1: Pre-linear (Input -> Hidden)
        x = nn.Dense(self.n_hidden, name="linear_pre")(x)
        x = self.activation(x)
        
        # Step 2: Hidden Layers (Hidden -> Hidden) with Residuals
        for i in range(self.n_layers):
            y = nn.Dense(self.n_hidden, name=f"linears_{i}")(x)
            y = self.activation(y)
            if self.residual:
                x = x + y  # Residual connection: x_{i+1} = Act(Linear(x_i)) + x_i
            else:
                x = y
                
        # Step 3: Post-linear (Hidden -> Output)
        x = nn.Dense(self.n_output, name="linear_post")(x)
        return x
