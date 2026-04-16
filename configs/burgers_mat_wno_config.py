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
class BurgersMatWNO1DConfig:
    """Configuration for WNO1D on Burgers dataset in .mat format."""
    train_path: str = "scirex/operators/data/burgers_data_512_51.mat"
    
    n_train: int = 400
    n_test: int = 100
    batch_size: int = 20
    epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    
    hidden_channels: int = 32
    n_layers: int = 4
    level: int = 4
    wavelet: str = "db4"
    mode: str = "symmetric"
    padding: float = 0.0
    lifting_channel_ratio: int = 2
    projection_channel_ratio: int = 2
    skip_type: str = "linear"
    
    encode_input: bool = True
    encode_output: bool = True
    seed: int = 42
    run_name: str = "burgers1d_wno"
