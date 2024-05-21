# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for loss function module."""

import unittest

import torch

from tests.helper import decorator
from msamp.common.dtype import Dtypes
from msamp.nn import LinearReplacer
from msamp.common.tensor import ScalingMeta
from msamp.operators.loss_fn import Loss_fn


class LossfnTestClass(unittest.TestCase):
    '''A class for Activation test cases.
    
    Args:
        unittest.TestCase (unittest.TestCase): TestCase class.
    '''
    def setUp(self):
        """Hook method for setting up the test fixture before exercising it."""
        torch.manual_seed(1000)

    def tearDown(self) -> None:
        """Hook method for deconstructing the test fixture after testing it."""
        pass
    
    @decorator.cuda_test
    def test_scalingSum(self):
        '''Test the function Loss_fn.sum().'''
        input = torch.randn((3, 4), dtype=torch.float16, device='cuda')
        linear = torch.nn.Linear(4, 8, bias=False).cuda().half()
        
        model1 = LinearReplacer.replace(linear, Dtypes.kfloat16)
        out1 = model1(input)
        sum1 = Loss_fn.sum(out1)
        self.assertTrue(sum1.requires_grad)
        
        model2 = LinearReplacer.replace(linear, Dtypes.kfloat16, enabling_fp8_activation=True)
        out2 = model2(input)
        self.assertTrue(out2.is_fp8_form == True)
        sum2 = Loss_fn.sum(out2)
        self.assertTrue(sum2.requires_grad)
        self.assertAlmostEqual(sum1.item(), sum2.item(), delta=1e-1)
        # under this case the delta cannot be set to 1e-2 or lower: -2.095703125 != -2.068359375 within 0.01 delta (0.02734375 difference)
        
        sum1.backward()
        sum2.backward()
        # Here must emphasize that, in most case, the gradients of the two models are very close, but not exactly the same.
        # In this special case, since the gradient back from Sum function is always 1, and we only have one gradient backward process, so the gradients of the two models are exactly the same.
        self.assertTrue(torch.equal(model1.weight.grad.float(), model2.weight.grad.float()))    
        self.assertTrue(torch.allclose(model1.weight.grad.float(), model2.weight.grad.float(), atol=1e-3))
        
        # python -m unittest tests.operators.test_loss.LossfnTestClass.test_scalingSum