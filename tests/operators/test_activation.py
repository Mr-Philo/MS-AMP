# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for activation module."""

import unittest
import copy

import torch

from tests.helper import decorator
from msamp.common.dtype import Dtypes
from msamp.nn import LinearReplacer
from msamp.operators.activation import Activation
from msamp.common.tensor import TypeCast, ScalingMeta


class ActivationTestClass(unittest.TestCase):
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
    def test_gelu(self):
        '''Test the function Activation.gelu().'''
        input = torch.ones((4, 4), dtype=torch.float16, device='cuda')
        linear = torch.nn.Linear(4, 8, bias=False).cuda().half()
        
        model1 = LinearReplacer.replace(linear, Dtypes.kfloat16)
        inner1 = model1(input)
        print(f"FP16 act model inner1: {inner1}, requires_grad: {inner1.requires_grad}")
        inner2 = Activation.gelu(inner1)
        print(f"FP16 act model inner2: {inner2}, requires_grad: {inner2.requires_grad}")
        
        model2 = LinearReplacer.replace(linear, Dtypes.kfloat16, enabling_fp8_activation=True)
        inner1 = model2(input)
        print(f"FP8 act model inner1: {inner1}, requires_grad: {inner1.requires_grad}")
        inner2 = Activation.gelu(inner1)       #! output_qtype = Dtypes.kfloat8_e4m3, supported
        print(f"FP8 act model inner2: {inner2}, requires_grad: {inner2.requires_grad}")
        casted_inner2 = TypeCast.cast_from_fp8(inner2.view(dtype=torch.uint8), inner2.scaling_meta, Dtypes.kfloat16) 

        ex_inner1 = linear(input)
        print(f"Expected model inner1: {ex_inner1}, requires_grad: {ex_inner1.requires_grad}")
        ex_inner2 = torch.nn.functional.gelu(ex_inner1)
        print(f"Expected model inner2: {ex_inner2}, requires_grad: {ex_inner2.requires_grad}")
        
        print(f"diff: {ex_inner2 - casted_inner2}")
        
        # python -m unittest tests.operators.test_activation.ActivationTestClass.test_gelu
        
    @decorator.cuda_test
    def test_sequential_gelu(self):
        '''Test the gelu() function in a sequantial module.'''
        input = torch.randn((3, 4), dtype=torch.float16, device='cuda')
        
        class MyModel(torch.nn.Module):
            def __init__(self):
                super(MyModel, self).__init__()
                self.linear1 = torch.nn.Linear(4, 8, bias=False).cuda()
                self.linear2 = torch.nn.Linear(8, 8, bias=False).cuda()
                self.linear3 = torch.nn.Linear(8, 4, bias=False).cuda()
                
            def forward(self, x):
                x = self.linear1(x)
                x = Activation.gelu(x)
                x = self.linear2(x)
                x = Activation.gelu(x)
                x = self.linear3(x)
                return x
        
        model = MyModel().cuda()
        model1 = copy.deepcopy(model)
        model1 = LinearReplacer.replace(model1, Dtypes.kfloat16)
        output1 = model1(input)
        print(f"fp16 model output: {output1}, with dtype: {output1.dtype}")
        
        print("------------For fp8 activation model------------")
        model2 = copy.deepcopy(model)
        model2 = LinearReplacer.replace(model2, Dtypes.kfloat16, enabling_fp8_activation=True)
        print(model2)
        output = model2(input)
        assert output.is_fp8_form == True
        print(f"output: {output}, with dtype: {output.dtype}, requires_grad: {output.requires_grad}, scaling_meta: {output.scaling_meta}")
        output = TypeCast.cast_from_fp8(output.view(dtype=torch.uint8), output.scaling_meta, Dtypes.kfloat16)
        print(f"casted output: {output}")
        
        print(f"difference: {output1 - output}")
        
        # python -m unittest tests.operators.test_activation.ActivationTestClass.test_sequential_gelu
    
    @decorator.cuda_test
    def test_gelu_backward(self):
        '''Test the backward of the function Activation.gelu().'''
        input = torch.randn((3, 4), dtype=torch.float16, device='cuda')
        linear = torch.nn.Linear(4, 8, bias=False).cuda()
        
        # for standard comparison
        print("------------For fp16 activation model------------")
        model1 = copy.deepcopy(linear)
        model1 = LinearReplacer.replace(model1, Dtypes.kfloat16)
        inner1 = model1(input)
        # inner1.retain_grad()
        print(f"FP16 act model inner: {inner1}, requires_grad: {inner1.requires_grad}")
        output1 = Activation.gelu(inner1)
        print(f"FP16 act model gelu output: {output1}, requires_grad: {output1.requires_grad}")
        loss = output1.sum()
        print(f"output1.sum: {loss}, with requires_grad: {loss.requires_grad}, with grad_fn: {loss.grad_fn}")
        loss.backward()
        print(f"fp16 model weight grad: {model1.weight.grad}, with dtype: {model1.weight.grad.dtype}, with requires_grad: {model1.weight.grad._requires_grad}")
        # print(f"check grad of inner1: {inner1.grad}")
        
        print("------------For fp8 activation model------------")
        model2 = copy.deepcopy(linear)   
        model2 = LinearReplacer.replace(model2, Dtypes.kfloat16, enabling_fp8_activation=True)
        inner2 = model2(input)
        assert inner2.is_fp8_form == True
        # inner2.retain_grad()
        print(f"FP8 model inner: {inner2}, with requires_grad: {inner2.requires_grad}")
        output2 = Activation.gelu(inner2)
        assert output2.is_fp8_form == True
        print(f"FP8 model gelu output: {output2}, with requires_grad: {output2.requires_grad}")
        
        #! we must make sure that the loss function contains the gradient
        class ScalingSum(torch.autograd.Function):
            @staticmethod
            def forward(ctx, input):
                meta = input.scaling_meta
                ctx.shape = input.shape     # torch.Size([3,2]), 这里是view前FP16 tensor的形状，为的是backward返回到形状不会出错（要和input的value形状保持一致）
                input = input.view(dtype=torch.uint8)
                ctx.true_out_shape = input.shape    # torch.Size([3,4]), 这里是view后FP8 tensor的形状
                input = TypeCast.cast_from_fp8(input, meta, Dtypes.kfloat32)
                return input.sum()
            
            @staticmethod
            def backward(ctx, grad_output):
                activation_grad = grad_output * torch.ones(ctx.true_out_shape, dtype=torch.float16).cuda()  # torch.Size([3,4])
                
                activation_grad_scaling_tensor = activation_grad.cast(Dtypes.kfloat8_e5m2, meta=ScalingMeta(Dtypes.kfloat8_e5m2))    # todo: 在这个SclingSum函数里面，只是为了传回去的activation_grad有scaling_meta这个属性，但是这里这个meta是没有用的，因为毕竟Sum函数的grad_fn是全一。如果之后要实现别的自定义损失函数，这里的meta就有用了，而且可能要考虑forward中的meta值
                activation_grad = activation_grad_scaling_tensor.value.view(dtype=torch.float16)      # torch.Size([3,4]) to torch.Size([3,2])
                activation_grad.scaling_meta = activation_grad_scaling_tensor.meta
                activation_grad.is_fp8_form = True
                
                assert activation_grad.shape == ctx.shape, f"Activation grad shape should be the same as input shape {ctx.shape}, but got {activation_grad.shape}"
                print(f"ScalingSum return: {activation_grad}")
                return activation_grad
            
        loss = ScalingSum.apply(output2)
        print(f"output2.sum: {loss}, with requires_grad: {loss.requires_grad}, with grad_fn: {loss.grad_fn}")
        loss.backward()
        print(f"weight grad: {model2.weight.grad}, with dtype: {model1.weight.grad.dtype}, with requires_grad: {model1.weight.grad._requires_grad}") 
        # print(f"check grad of inner2: {inner2.grad}")
        
        assert model2.weight.shape == model2.weight.grad.shape, f"Weight grad shape should be the same as model weight shape {model2.weight.shape}, but got {model2.weight.grad.shape}"
        
        # python -m unittest tests.operators.test_activation.ActivationTestClass.test_gelu_backward
        
    def test_relu(self):
        '''Test the function Activation.relu().'''
        pass
    