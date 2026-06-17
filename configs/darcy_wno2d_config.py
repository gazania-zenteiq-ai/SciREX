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

"""Experiment config for WNO2D on the original Darcy MATLAB benchmark."""

import os
from dataclasses import dataclass


def _default_darcy_path(filename: str) -> str:
    """Prefer local Darcy copies before falling back to repo data folders."""
    home = os.path.expanduser("~")
    candidates = [
        os.path.join(home, "Harshdeep", "Data", "Darcy_421", filename),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return candidates[0]


@dataclass
class DarcyWNO2DConfig:
    # Dataset
    train_path: str = _default_darcy_path("piececonst_r421_N1024_smooth1.mat")
    test_path: str = _default_darcy_path("piececonst_r421_N1024_smooth2.mat")
    n_train: int = 1000
    n_test: int = 100
    subsample_rate: int = 5

    # Model
    hidden_channels: int = 64
    n_layers: int = 4
    level: int = 4
    wavelet: str = "db4"
    mode: str = "symmetric"
    padding: float = 0.0
    lifting_channel_ratio: int = 2
    projection_channel_ratio: int = 2
    skip_type: str = "linear"

    # Training
    batch_size: int = 20
    epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    encode_input: bool = True
    encode_output: bool = True
    seed: int = 42

    # Outputs
    run_name: str = "darcy2d_wno"
