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
Experiment configurations for FNO on Poisson equation (2D and 3D).

These configs compose a *model preset* from ``configs.models`` with
experiment-specific training and data-generation parameters.

Usage
-----
    from configs.poisson_fno_config import FNO2DConfig, FNO3DConfig

    config = FNO2DConfig()
    # config.model   → FNO_Medium2D instance  (architecture params)
    # config.*       → training / data params  (lr, batch_size, …)
"""

from typing import Literal, List
from zencfg import ConfigBase
from configs.models import FNO_Medium2D, FNO_Medium3D, FNO_Large2D

class PoissonOptConfig(ConfigBase):
    learning_rate: float = 5e-3
    weight_decay: float = 1e-4
    epochs: int = 500
    steps_per_epoch: int = 10
    scheduler_type: Literal["step", "cosine"] = "cosine"
    cosine_decay_epochs: int = 500
    scheduler_step_size: int = 100
    scheduler_gamma: float = 0.5

class PoissonDatasetConfig(ConfigBase):
    batch_size: int = 32
    n_test: int = 200
    n_train: int = 2000
    resolution: List[int] = [64, 64]
    seed: int = 42


class FNO2DConfig(ConfigBase):
    """Experiment config for 2D Poisson using FNO."""

    model: FNO_Medium2D = FNO_Medium2D()
    opt: PoissonOptConfig = PoissonOptConfig()
    data: PoissonDatasetConfig = PoissonDatasetConfig()


class Poisson3DOptConfig(PoissonOptConfig):
    learning_rate: float = 1e-3
    steps_per_epoch: int = 50
    scheduler_step_size: int = 30


class Poisson3DDatasetConfig(PoissonDatasetConfig):
    batch_size: int = 10
    n_test: int = 20
    resolution: List[int] = [32, 32, 32]
    include_mesh: bool = False


class FNO3DConfig(ConfigBase):
    """Experiment config for 3D Poisson using FNO."""

    model: FNO_Medium3D = FNO_Medium3D()
    opt: PoissonOptConfig = Poisson3DOptConfig()
    data: PoissonDatasetConfig = Poisson3DDatasetConfig()


