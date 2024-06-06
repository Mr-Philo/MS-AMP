# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Exposes the interface of MS-AMP activation function module."""

from msamp.operators.activation.activation import Activation, ScalingGelu, ScalingLayerNorm, ScalingDropout

__all__ = ['Activation', 'ScalingGelu', 'ScalingLayerNorm', 'ScalingDropout']