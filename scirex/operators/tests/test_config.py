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

"""Test that config presets instantiate with expected defaults."""

from configs.models import FNOConfig, FNO_Medium2D, FNO_Medium3D


def test_fno_config_defaults():
    config = FNOConfig()
    assert config.arch == "fno"
    assert config.hidden_channels == 64
    assert config.n_layers == 4
    assert config.use_grid is True


def test_fno_medium_2d_preset():
    config = FNO_Medium2D()
    assert config.n_modes == (24, 24)
    assert config.hidden_channels == 128
    assert config.use_norm is True


def test_fno_medium_3d_preset():
    config = FNO_Medium3D()
    assert config.n_modes == (16, 16, 16)
    assert config.hidden_channels == 128
    assert config.use_norm is True
