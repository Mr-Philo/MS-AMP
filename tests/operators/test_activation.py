# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for activation module."""

import unittest

import torch

from tests.helper import decorator
from msamp.common.dtype import Dtypes
from msamp.nn import LinearReplacer
from msamp.operators.activation import Activation


class ActivationTestClass(unittest.TestCase):
    '''A class for Activation test cases.
    
    Args:
        unittest.TestCase (unittest.TestCase): TestCase class.
    '''
    @decorator.cuda_test
    def test_gelu(self):
        '''Test the function Activation.gelu().'''
        input = torch.ones((4, 4), dtype=torch.float32, device='cuda')
        linear = torch.nn.Linear(4, 8, bias=False).cuda()
        
        model1 = LinearReplacer.replace(linear, Dtypes.kfloat16)
        inner1 = model1(input)
        print(f"FP16 act model inner1: {inner1}, requires_grad: {inner1.requires_grad}")
        inner2 = Activation.gelu(inner1, Dtypes.kfloat16)
        print(f"FP16 act model inner2: {inner2}, requires_grad: {inner2.requires_grad}")
        
        model2 = LinearReplacer.replace(linear, Dtypes.kfloat16, enabling_fp8_activation=True)
        inner1 = model2(input)
        print(f"FP8 act model inner1: {inner1}, requires_grad: {inner1.requires_grad}")
        inner2 = Activation.gelu(inner1, Dtypes.kfloat16)
        print(f"FP8 act model inner2: {inner2}, requires_grad: {inner2.requires_grad}")

        ex_inner1 = linear(input)
        print(f"Expected model inner1: {ex_inner1}, requires_grad: {ex_inner1.requires_grad}")
        ex_inner2 = torch.nn.functional.gelu(ex_inner1)
        print(f"Expected model inner2: {ex_inner2}, requires_grad: {ex_inner2.requires_grad}")
        
        print(f"diff: {ex_inner2 - inner2}")
        
        # python -m unittest tests.operators.test_activation.ActivationTestClass.test_gelu
        
    def test_relu(self):
        '''Test the function Activation.relu().'''
        pass
    