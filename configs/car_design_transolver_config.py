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

from ml_collections import config_dict
import optax
from flax import linen as nn

def get_config() -> config_dict.ConfigDict:
    """Configuration for training Transolver on Car-Design-ShapeNetCar."""
    config = config_dict.ConfigDict()

    # Data configuration
    config.data = config_dict.ConfigDict()
    config.data.data_dir = "/home/harshdeep/Harshdeep/Data/mlcfd_data/training_data"
    config.data.batch_size = 1            # matches reference (batch_size=1)
    config.data.shuffle = True
    config.data.fold_id = 0               # param folder held out as validation (0..8)

    # Model configuration (Transolver) — matches reference main.py exactly
    # Input: space_dim=7 → [x, y, z, sdf, nx, ny, nz] per point
    config.model = config_dict.ConfigDict()
    config.model.hidden_channels = 256
    config.model.out_channels = 4       # [velo_x, velo_y, velo_z, pressure]
    config.model.n_layers = 8
    config.model.n_heads = 8
    config.model.dim_head = 32          # 256 / 8 = 32
    config.model.slice_num = 32
    config.model.mlp_ratio = 2.0
    config.model.dropout = 0.0

    # Training configuration
    config.training = config_dict.ConfigDict()
    config.training.num_epochs = 200
    config.training.learning_rate = 1e-3
    config.training.weight_decay = 1e-4
    config.training.reg_weight = 0.5    # Weight for pressure loss on surface
    config.training.val_iter = 5        # Evaluate on val split every N epochs

    # Optimizer configuration
    config.optimizer = config_dict.ConfigDict()
    config.optimizer.name = "adamw"     # AdamW is generally preferred
    
    return config
