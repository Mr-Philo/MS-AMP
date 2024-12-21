# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# added ruizhe 2024/12/20

"""Exposes the interface of MS-AMP GEMM module."""

from msamp.operators.fp4_quant.fp4_quant import FP4_QUANT

__all__ = ['FP4_QUANT']