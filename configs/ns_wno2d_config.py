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

from dataclasses import dataclass

@dataclass
class NSMatWNO2DConfig:
    """Configuration for WNO2D on Navier-Stokes dataset in .mat format."""
    train_path: str = "scirex/operators/data/NavierStokes_V1e-3_N5000_T50/ns_V1e-3_N5000_T50.mat"
    
    n_train: int = 1000
    n_test: int = 100
    t_in: int = 10          # Number of input time steps
    t_out: int = 10         # Number of output predicting time steps
    batch_size: int = 20
    epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    
    hidden_channels: int = 64
    n_layers: int = 4
    level: int = 4          # Wavelet level
    wavelet: str = "db6"
    mode: str = "symmetric"
    padding: float = 0.0
    lifting_channel_ratio: int = 2
    projection_channel_ratio: int = 2
    skip_type: str = "linear"
    
    encode_input: bool = True
    encode_output: bool = True
    seed: int = 42
    run_name: str = "ns2d_wno"
