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

import math
from functools import lru_cache
from itertools import product
from typing import Dict, List, Sequence, Tuple, Union

import jax
import jax.numpy as jnp
import jaxwt
from flax import linen as nn

_SUPPORTED_MODES = {
    "symmetric": "symmetric",
    "reflect": "reflect",
    "zero": "zero",
    "constant": "zero",
}


def _normalize_size(size: Union[int, Sequence[int]], ndim: int) -> Tuple[int, ...]:
    if isinstance(size, int):
        return (size,) * ndim

    size_tuple = tuple(size)
    if len(size_tuple) != ndim:
        raise ValueError(f"Expected size with {ndim} entries, got {size_tuple}.")
    return size_tuple


def _normalize_mode(mode: str) -> str:
    normalized = _SUPPORTED_MODES.get(mode.lower())
    if normalized is None:
        raise ValueError(
            f"Unsupported wavelet boundary mode '{mode}'. "
            "jaxwt supports 'symmetric', 'reflect', and 'zero'."
        )
    return normalized


def _resolution_factor(input_size: Tuple[int, ...], base_size: Tuple[int, ...]) -> int:
    def _floor_log2_ratio(numerator: int, denominator: int) -> int:
        ratio = max(1, numerator // denominator)
        return ratio.bit_length() - 1

    if all(inp == base for inp, base in zip(input_size, base_size)):
        return 0

    if all(inp >= base for inp, base in zip(input_size, base_size)):
        return min(
            _floor_log2_ratio(inp, base) for inp, base in zip(input_size, base_size)
        )

    if all(inp <= base for inp, base in zip(input_size, base_size)):
        return -min(
            _floor_log2_ratio(base, inp) for inp, base in zip(input_size, base_size)
        )

    return 0


def _effective_level_wno(
    input_size: Tuple[int, ...], base_size: Tuple[int, ...], level: int
) -> int:
    def _floor_log2_ratio(numerator: int, denominator: int) -> int:
        ratio = max(1, numerator // denominator)
        return ratio.bit_length() - 1

    ref_input = input_size[-1]
    ref_base = base_size[-1]
    factor = 0
    if ref_input > ref_base:
        factor = _floor_log2_ratio(ref_input, ref_base)
    elif ref_input < ref_base:
        factor = -_floor_log2_ratio(ref_base, ref_input)
    return max(0, level + factor)


def _subband_codes(ndim: int) -> List[Tuple[int, ...]]:
    return list(product((0, 1), repeat=ndim))


def _subband_name(code: Tuple[int, ...], ndim: int) -> str:
    if ndim == 1:
        return "weights_approx" if code == (0,) else "weights_detail"
    if ndim == 2:
        labels = {0: "l", 1: "h"}
        return "w_" + "".join(labels[bit] for bit in code)
    return "w_" + "".join("l" if bit == 0 else "h" for bit in code)


def _apply_subband_weight(coeff: jnp.ndarray, weight: jnp.ndarray) -> jnp.ndarray:
    spatial_shape = coeff.shape[1:-1]
    weight_shape = weight.shape[2:]
    active_shape = tuple(
        min(cdim, wdim) for cdim, wdim in zip(spatial_shape, weight_shape)
    )

    coeff_slices = (slice(None),) + tuple(
        slice(0, dim) for dim in active_shape
    ) + (slice(None),)
    weight_slices = (slice(None), slice(None)) + tuple(
        slice(0, dim) for dim in active_shape
    )

    coeff_active = coeff[coeff_slices]
    weight_active = weight[weight_slices]
    batch = coeff.shape[0]
    in_channels = coeff_active.shape[-1]
    out_channels = weight_active.shape[1]
    flat_modes = math.prod(active_shape)

    coeff_flat = coeff_active.reshape(batch, flat_modes, in_channels)
    weight_flat = weight_active.reshape(in_channels, out_channels, flat_modes)
    out_flat = jnp.einsum("bmi,iom->bmo", coeff_flat, weight_flat)
    out_active = out_flat.reshape((batch,) + active_shape + (out_channels,))

    out = jnp.zeros((batch,) + spatial_shape + (out_channels,), dtype=coeff.dtype)
    out_slices = (slice(None),) + tuple(slice(0, dim) for dim in active_shape) + (
        slice(None),
    )
    return out.at[out_slices].set(out_active)


def _zeros_with_output_channels(coeff: jnp.ndarray, out_channels: int) -> jnp.ndarray:
    return jnp.zeros(coeff.shape[:-1] + (out_channels,), dtype=coeff.dtype)


def _pad_to_power_of_two(
    x: jnp.ndarray, input_size: Tuple[int, ...], effective_level: int, mode: str
) -> Tuple[jnp.ndarray, Tuple[int, ...]]:
    scale_factor = 2**effective_level
    target_size = tuple(
        ((dim - 1) // scale_factor + 1) * scale_factor for dim in input_size
    )
    pad_sizes = tuple(
        target - current for target, current in zip(target_size, input_size)
    )

    if any(pad_sizes):
        x = jnp.pad(
            x,
            ((0, 0),) + tuple((0, pad) for pad in pad_sizes) + ((0, 0),),
            mode=mode,
        )

    return x, pad_sizes


def _crop_spatial(x: jnp.ndarray, target_shape: Tuple[int, ...]) -> jnp.ndarray:
    slices = (slice(None),) + tuple(slice(0, dim) for dim in target_shape) + (
        slice(None),
    )
    return x[slices]


def _to_jaxwt_layout(x: jnp.ndarray, ndim: int) -> jnp.ndarray:
    if ndim == 1:
        return jnp.transpose(x, (0, 2, 1))
    if ndim == 2:
        return jnp.transpose(x, (0, 3, 1, 2))
    if ndim == 3:
        return jnp.transpose(x, (0, 4, 1, 2, 3))
    raise ValueError("WaveletConv with jaxwt currently supports only 1D, 2D, and 3D inputs.")


def _from_jaxwt_array(x: jnp.ndarray, ndim: int) -> jnp.ndarray:
    if ndim == 1:
        return jnp.transpose(x, (0, 2, 1))
    if ndim == 2:
        return jnp.transpose(x, (0, 2, 3, 1))
    if ndim == 3:
        return jnp.transpose(x, (0, 2, 3, 4, 1))
    raise ValueError("WaveletConv with jaxwt currently supports only 1D, 2D, and 3D inputs.")


def _coeffs_from_jaxwt_layout(coeffs, ndim: int):
    if ndim == 1:
        return [_from_jaxwt_array(coeff, ndim) for coeff in coeffs]
    if ndim == 2:
        converted = [_from_jaxwt_array(coeffs[0], ndim)]
        converted.extend(
            tuple(_from_jaxwt_array(part, ndim) for part in detail) for detail in coeffs[1:]
        )
        return converted
    converted = [_from_jaxwt_array(coeffs[0], ndim)]
    converted.extend(
        {key: _from_jaxwt_array(value, ndim) for key, value in detail.items()}
        for detail in coeffs[1:]
    )
    return converted


def _coeffs_to_jaxwt_layout(coeffs, ndim: int):
    if ndim == 1:
        return [_to_jaxwt_layout(coeff, ndim) for coeff in coeffs]
    if ndim == 2:
        converted = [_to_jaxwt_layout(coeffs[0], ndim)]
        converted.extend(
            tuple(_to_jaxwt_layout(part, ndim) for part in detail) for detail in coeffs[1:]
        )
        return converted
    converted = [_to_jaxwt_layout(coeffs[0], ndim)]
    converted.extend(
        {key: _to_jaxwt_layout(value, ndim) for key, value in detail.items()}
        for detail in coeffs[1:]
    )
    return converted


def _wavedec_nd(x: jnp.ndarray, wavelet: str, mode: str, level: int, ndim: int):
    x_cf = _to_jaxwt_layout(x, ndim)
    if ndim == 1:
        coeffs = jaxwt.wavedec(x_cf, wavelet, mode=mode, level=level)
        return _coeffs_from_jaxwt_layout(coeffs, ndim)
    if ndim == 2:
        coeffs = jaxwt.wavedec2(x_cf, wavelet, mode=mode, level=level)
        return _coeffs_from_jaxwt_layout(coeffs, ndim)
    if ndim == 3:
        coeffs = jaxwt.wavedec3(x_cf, wavelet, mode=mode, level=level)
        return _coeffs_from_jaxwt_layout(coeffs, ndim)
    raise ValueError("WaveletConv with jaxwt currently supports only 1D, 2D, and 3D inputs.")


def _waverec_nd(coeffs, wavelet: str, ndim: int) -> jnp.ndarray:
    coeffs_cf = _coeffs_to_jaxwt_layout(coeffs, ndim)
    if ndim == 1:
        return _from_jaxwt_array(jaxwt.waverec(coeffs_cf, wavelet), ndim)
    if ndim == 2:
        return _from_jaxwt_array(jaxwt.waverec2(coeffs_cf, wavelet), ndim)
    if ndim == 3:
        return _from_jaxwt_array(jaxwt.waverec3(coeffs_cf, wavelet), ndim)
    raise ValueError("WaveletConv with jaxwt currently supports only 1D, 2D, and 3D inputs.")


@lru_cache(maxsize=128)
def _mode_sizes(
    size: Tuple[int, ...], level: int, wavelet: str, mode: str, ndim: int
) -> Tuple[int, ...]:
    if level == 0:
        return size

    dummy = jnp.zeros((1,) + size + (1,), dtype=jnp.float32)
    coeffs = _wavedec_nd(dummy, wavelet, mode, level, ndim)
    return tuple(coeffs[0].shape[1:-1])


def _detail_from_coeffs(coeffs, ndim: int) -> Dict[Tuple[int, ...], jnp.ndarray]:
    if ndim == 1:
        return {(1,): coeffs[1]}
    if ndim == 2:
        c_lh, c_hl, c_hh = coeffs[1]
        return {(0, 1): c_lh, (1, 0): c_hl, (1, 1): c_hh}
    coeff_dict = coeffs[1]
    return {
        (0, 0, 1): coeff_dict["aad"],
        (0, 1, 0): coeff_dict["ada"],
        (0, 1, 1): coeff_dict["add"],
        (1, 0, 0): coeff_dict["daa"],
        (1, 0, 1): coeff_dict["dad"],
        (1, 1, 0): coeff_dict["dda"],
        (1, 1, 1): coeff_dict["ddd"],
    }


def _zero_detail_like(detail, out_channels: int, ndim: int):
    if ndim == 1:
        return _zeros_with_output_channels(detail, out_channels)
    if ndim == 2:
        return tuple(_zeros_with_output_channels(part, out_channels) for part in detail)
    return {
        key: _zeros_with_output_channels(value, out_channels)
        for key, value in detail.items()
    }


def _weighted_detail(
    detail_map: Dict[Tuple[int, ...], jnp.ndarray], weights, ndim: int
):
    if ndim == 1:
        return _apply_subband_weight(detail_map[(1,)], weights[(1,)])
    if ndim == 2:
        return (
            _apply_subband_weight(detail_map[(0, 1)], weights[(0, 1)]),
            _apply_subband_weight(detail_map[(1, 0)], weights[(1, 0)]),
            _apply_subband_weight(detail_map[(1, 1)], weights[(1, 1)]),
        )
    return {
        "aad": _apply_subband_weight(detail_map[(0, 0, 1)], weights[(0, 0, 1)]),
        "ada": _apply_subband_weight(detail_map[(0, 1, 0)], weights[(0, 1, 0)]),
        "add": _apply_subband_weight(detail_map[(0, 1, 1)], weights[(0, 1, 1)]),
        "daa": _apply_subband_weight(detail_map[(1, 0, 0)], weights[(1, 0, 0)]),
        "dad": _apply_subband_weight(detail_map[(1, 0, 1)], weights[(1, 0, 1)]),
        "dda": _apply_subband_weight(detail_map[(1, 1, 0)], weights[(1, 1, 0)]),
        "ddd": _apply_subband_weight(detail_map[(1, 1, 1)], weights[(1, 1, 1)]),
    }


class WaveletConv(nn.Module):
    """
    N-dimensional Wavelet Convolution layer following the WNO architecture.

    The decomposition and reconstruction are delegated to `jaxwt`, while
    SciREX keeps the learnable wavelet-domain mixing consistent with the rest
    of the WNO stack.
    """

    in_channels: int
    out_channels: int
    level: int = 1
    size: Union[int, Tuple[int, ...]] = 1024
    wavelet: str = "db4"
    mode: str = "symmetric"

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        ndim = x.ndim - 2
        if ndim < 1:
            raise ValueError("WaveletConv expects at least one spatial dimension.")
        if x.shape[-1] != self.in_channels:
            raise ValueError(
                f"Input channels mismatch: expected {self.in_channels}, got {x.shape[-1]}."
            )

        normalized_mode = _normalize_mode(self.mode)
        training_size = _normalize_size(self.size, ndim)
        input_size = tuple(x.shape[1:-1])
        effective_level = _effective_level_wno(input_size, training_size, self.level)
        x, pad_sizes = _pad_to_power_of_two(
            x, input_size, effective_level, normalized_mode
        )

        subband_codes = _subband_codes(ndim)
        approx_code = (0,) * ndim
        mode_sizes = _mode_sizes(
            training_size, self.level, self.wavelet, normalized_mode, ndim
        )

        scale = 1.0 / (self.in_channels * self.out_channels)
        init_fn = lambda key, shape: scale * jax.random.uniform(key, shape)
        weights = {
            code: self.param(
                _subband_name(code, ndim),
                init_fn,
                (self.in_channels, self.out_channels) + mode_sizes,
            )
            for code in subband_codes
        }

        if effective_level == 0:
            res = _apply_subband_weight(x, weights[approx_code])
            return _crop_spatial(res, input_size) if any(pad_sizes) else res

        coeffs = _wavedec_nd(x, self.wavelet, normalized_mode, effective_level, ndim)
        weighted_approx = _apply_subband_weight(coeffs[0], weights[approx_code])
        weighted_detail = _weighted_detail(_detail_from_coeffs(coeffs, ndim), weights, ndim)

        recon_coeffs = [weighted_approx, weighted_detail]
        for detail in coeffs[2:]:
            recon_coeffs.append(_zero_detail_like(detail, self.out_channels, ndim))

        res = _waverec_nd(recon_coeffs, self.wavelet, ndim)

        if any(pad_sizes):
            res = _crop_spatial(res, input_size)

        return res
