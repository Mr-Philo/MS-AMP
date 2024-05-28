# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for loss function module."""

import unittest

import torch

from tests.helper import decorator
from msamp.common.dtype import Dtypes
from msamp.nn import LinearReplacer
from msamp.common.tensor import ScalingMeta
from msamp.operators.activation import Activation
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
        
    @decorator.cuda_test
    def test_scalingNLLloss(self):
        '''Test the function Loss_fn.nll_loss().'''  
        # input = torch.randn((3, 4), dtype=torch.float16, device='cuda', requires_grad=True)   #! here cannot set requires_grad=True, since FP8GemmFunction has not yet support grad computing for the very first input tensor
        input = torch.randn((3, 4), dtype=torch.float16, device='cuda')
        target = torch.randint(0, 4, (3,), dtype=torch.long, device='cuda')  
  
        linear = torch.nn.Linear(4, 8, bias=False).cuda().half()  
        
        model1 = LinearReplacer.replace(linear, Dtypes.kfloat16)  
        inner1 = model1(input)  
        out1 = Activation.log_softmax(inner1, dim=1)
        loss1 = Loss_fn.nll_loss(out1, target)  
        self.assertTrue(loss1.requires_grad)  
          
        model2 = LinearReplacer.replace(linear, Dtypes.kfloat16, enabling_fp8_activation=True)  
        inner2 = model2(input)  
        out2 = Activation.log_softmax(inner2, dim=1)
        self.assertTrue(out2.is_fp8_form == True)  
        loss2 = Loss_fn.nll_loss(out2, target)  
        self.assertTrue(loss2.requires_grad)  
          
        # print(f"loss1: {loss1}, grad_fn: {loss1.grad_fn}")
        # print(f"loss2: {loss2}, grad_fn: {loss2.grad_fn}")
        self.assertAlmostEqual(loss1.item(), loss2.item(), delta=5e-2)  
          
        # Backward pass  
        loss1.backward()  
        loss2.backward()  
          
        # Check that the gradients are very close  
        # print(f"model1.weight.grad: {model1.weight.grad}")
        # print(f"model2.weight.grad: {model2.weight.grad}")
        self.assertTrue(torch.allclose(model1.weight.grad.float(), model2.weight.grad.float(), atol=5e-2))  
          
        # Clean up for next test  
        model1.weight.grad = None  
        model2.weight.grad = None 
        
        # python -m unittest tests.operators.test_loss.LossfnTestClass.test_scalingNLLloss