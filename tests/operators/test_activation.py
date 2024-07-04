# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for activation module."""

import unittest
import copy

import torch
import torch.nn.grad

from tests.helper import decorator
from msamp.common.dtype import Dtypes
from msamp.nn import LinearReplacer
from msamp.operators.activation import Activation, ActivationReplacer
from msamp.common.tensor import TypeCast, ScalingMeta
from msamp.operators.loss_fn import Loss_fn


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
    
    @staticmethod
    def standard_sequential_model_valid(model):
        '''Test the standard sequential model with fp16 and fp8 activation.'''
        
        # input = torch.randn((3, 4, 4), dtype=torch.float16, device='cuda')
        input = torch.randn((3, 4, 4), device='cuda')      # todo: if need fp16

        torch.manual_seed(42)
        model1 = copy.deepcopy(model)
        model1 = LinearReplacer.replace(model1, Dtypes.kfloat16)
        output1 = model1(input)
        
        torch.manual_seed(42)
        model2 = copy.deepcopy(model)
        model2 = LinearReplacer.replace(model2, Dtypes.kfloat16, enabling_fp8_activation=True)
        output2 = model2(input)
        
        assert output2.is_fp8_form == True
        assert torch.allclose(output1, TypeCast.cast_from_fp8_activation(output2), atol=0.12)
        # for Scaling LayerNorm, the atol will be larger than 0.1, but not so large.
        
        # backward
        loss1 = Loss_fn.sum(output1)
        loss1.backward()
        layer = model1.linear1 if hasattr(model1, 'linear1') else model1.linear
        grad1 = layer.weight.grad
        loss2 = Loss_fn.sum(output2)
        loss2.backward()
        grad2 = layer.weight.grad
        
        assert layer.weight.grad.shape == layer.weight.grad.shape
        assert torch.allclose(grad1.float(), grad2.float(), atol=1e-3)
       
        
    @staticmethod
    def standard_sequential_model_valid_with_info_printing(model):
        '''Test the standard sequential model with fp16 and fp8 activation, and print the information for better debugging'''
        
        input = torch.randn((3, 4, 4), device='cuda')
        # input = torch.randn((3, 4, 4), dtype=float16, device='cuda')      # todo: if need fp16
        
        print("------------For fp16 activation model------------")
        torch.manual_seed(42)
        model1 = copy.deepcopy(model)
        model1 = LinearReplacer.replace(model1, Dtypes.kfloat16)
        print(model1)
        output1 = model1(input)
        print(f"fp16 model output: {output1}, with dtype: {output1.dtype}, requires_grad: {output1.requires_grad}")
        
        print("------------For fp8 activation model------------")
        torch.manual_seed(42)
        model2 = copy.deepcopy(model)
        model2 = LinearReplacer.replace(model2, Dtypes.kfloat16, enabling_fp8_activation=True)
        print(model2)
        output2 = model2(input)
        assert output2.is_fp8_form == True
        print(f"fp8 model output: {output2}, with dtype: {output2.dtype}, requires_grad: {output2.requires_grad}, scaling_meta: {output2.scaling_meta}")        
        print(f"difference: {output1 - TypeCast.cast_from_fp8_activation(output2)}")
        
        # backward
        loss1 = Loss_fn.sum(output1)
        loss1.backward()
        layer = model1.linear1 if hasattr(model1, 'linear1') else model1.linear
        print(f"fp16 model weight grad: {layer.weight.grad}, with dtype: {layer.weight.grad.dtype}, with requires_grad: {layer.weight.grad._requires_grad}")
        loss2 = Loss_fn.sum(output2)
        loss2.backward()
        print(f"fp8 model weight grad: {layer.weight.grad}, with dtype: {layer.weight.grad.dtype}, with requires_grad: {layer.weight.grad._requires_grad}")
        
        assert layer.weight.grad.shape == layer.weight.grad.shape, f"Weight grad shape should be the same as model weight shape {layer.weight.grad.shape}, but got {layer.weight.grad.shape}"
        
        
    @decorator.cuda_test
    def test_module_gelu(self):
        '''Test ScalingGelu module.'''
        
        class MyModel(torch.nn.Module):
            def __init__(self):
                super(MyModel, self).__init__()
                self.linear = torch.nn.Linear(4, 8, bias=False)
                self.gelu = torch.nn.GELU()
                
            def forward(self, x):
                x = self.linear(x)
                x = self.gelu(x)
                return x
            
        model = MyModel().cuda()
        model = ActivationReplacer.replace(model)
        ActivationTestClass.standard_sequential_model_valid(model)
        # ActivationTestClass.standard_sequential_model_valid_with_info_printing(model)
        
        # python -m unittest tests.operators.test_activation.ActivationTestClass.test_module_gelu
        
        
    @decorator.cuda_test
    def test_sequential_relu_and_backward(self):
        '''Test the relu() function (fwd+bwd) in a sequantial module.'''
        
        class MyModel(torch.nn.Module):
            def __init__(self):
                super(MyModel, self).__init__()
                self.linear1 = torch.nn.Linear(4, 8, bias=False).cuda()
                self.linear2 = torch.nn.Linear(8, 8, bias=False).cuda()
                self.linear3 = torch.nn.Linear(8, 4, bias=False).cuda()
                
            def forward(self, x):
                x = self.linear1(x)
                x = Activation.relu(x)
                x = self.linear2(x)
                x = Activation.relu(x)
                x = self.linear3(x)
                return x
        
        model = MyModel().cuda()
        # model = ActivationReplacer.replace(model)     # ReLU not replaced in ActivationReplacer now
        ActivationTestClass.standard_sequential_model_valid(model)
        # ActivationTestClass.standard_sequential_model_valid_with_info_printing(model)
        
        # python -m unittest tests.operators.test_activation.ActivationTestClass.test_sequential_relu_and_backward
        
    @decorator.cuda_test
    def test_sequential_dropout_and_backward(self):
        '''Test the dropout() function (fwd+bwd) in a sequantial module.'''
        
        #! 这部分dropout的实现是不需要cast-compute-cast框架的
        
        input = torch.randn((3, 4), dtype=torch.float16, device='cuda')
        
        class MyModel(torch.nn.Module):
            def __init__(self):
                super(MyModel, self).__init__()
                self.linear1 = torch.nn.Linear(4, 8, bias=False)
                self.linear2 = torch.nn.Linear(8, 8, bias=False)
                self.linear3 = torch.nn.Linear(8, 4, bias=False)
                
            def forward(self, x):
                x = self.linear1(x)
                torch.manual_seed(42)
                # print(f"x before dropout: {TypeCast.cast_from_fp8_activation(x) if x.is_fp8_form else x}, meta: {x.scaling_meta}")
                x = Activation.dropout(x, 0.5)
                # print(f"x after dropout: {TypeCast.cast_from_fp8_activation(x) if x.is_fp8_form else x}, meta: {x.scaling_meta}")
                x = self.linear2(x)
                torch.manual_seed(42)
                x = Activation.dropout(x, 0.5)
                x = self.linear3(x)
                return x
        
        model = MyModel().cuda()
        model1 = copy.deepcopy(model)
        model1 = LinearReplacer.replace(model1, Dtypes.kfloat16)
        output1 = model1(input)
        # print(f"fp16 model output: {output1}, with dtype: {output1.dtype}")
        
        # print("------------For fp8 activation model------------")
        model2 = copy.deepcopy(model)
        model2 = LinearReplacer.replace(model2, Dtypes.kfloat16, enabling_fp8_activation=True)
        # print(model2)
        output2 = model2(input)
        assert output2.is_fp8_form == True
        # print(f"output: {output2}, with dtype: {output2.dtype}, requires_grad: {output2.requires_grad}, scaling_meta: {output2.scaling_meta}")
        output2_fp = TypeCast.cast_from_fp8(output2.view(dtype=torch.uint8), output2.scaling_meta, Dtypes.kfloat16)
        # print(f"casted output: {output2_fp}")
        
        # print(f"difference: {output1 - output2_fp}")
        # 虽然两次dropout前都set了随机种子，但由于底层实现逻辑还是不一样的，所以dropout mask的随机性会导致这里的计算结果并不一样
        
        # backward
        loss1 = Loss_fn.sum(output1)
        loss1.backward()
        # print(f"fp16 model weight grad: {model1.linear1.weight.grad}, with dtype: {model1.linear1.weight.grad.dtype}, with requires_grad: {model1.linear1.weight.grad._requires_grad}")
        loss2 = Loss_fn.sum(output2)
        loss2.backward()
        # print(f"fp8 model weight grad: {model2.linear1.weight.grad}, with dtype: {model2.linear1.weight.grad.dtype}, with requires_grad: {model2.linear1.weight.grad._requires_grad}")
        
        assert model2.linear1.weight.grad.shape == model2.linear1.weight.grad.shape, f"Weight grad shape should be the same as model weight shape {model2.linear1.weight.grad.shape}, but got {model2.linear1.weight.grad.shape}"
        
        # python -m unittest tests.operators.test_activation.ActivationTestClass.test_sequential_dropout_and_backward
    
    @decorator.cuda_test
    def test_sequential_logsoftmax_and_backward(self):
        '''Test the log_softmax() function (fwd+bwd) in a sequantial module.'''
        
        class MyModel(torch.nn.Module):
            def __init__(self):
                super(MyModel, self).__init__()
                self.linear1 = torch.nn.Linear(4, 8, bias=False)
                self.linear2 = torch.nn.Linear(8, 8, bias=False)
                self.linear3 = torch.nn.Linear(8, 4, bias=False)
                
            def forward(self, x):
                x = self.linear1(x)
                x = self.linear2(x)
                x = self.linear3(x)
                x = Activation.log_softmax(x, dim=1)
                return x
        
        model = MyModel().cuda()
        # model = ActivationReplacer.replace(model)     # Log_softmax not replaced in ActivationReplacer now
        ActivationTestClass.standard_sequential_model_valid(model)
        # ActivationTestClass.standard_sequential_model_valid_with_info_printing(model)
        
        # python -m unittest tests.operators.test_activation.ActivationTestClass.test_sequential_logsoftmax_and_backward
        
    @decorator.cuda_test
    def test_sequential_flatten_and_backward(self):
        '''Test the flatten() function (fwd+bwd) in a sequantial module.'''
        
        class MyModel(torch.nn.Module):
            def __init__(self):
                super(MyModel, self).__init__()
                self.linear1 = torch.nn.Linear(4, 8, bias=False)
                self.linear2 = torch.nn.Linear(32, 4, bias=False)
                
            def forward(self, x):
                x = self.linear1(x)
                x = Activation.flatten(x, 1)
                x = self.linear2(x)
                return x
        
        model = MyModel().cuda()
        # model = ActivationReplacer.replace(model)     # Flatten not replaced in ActivationReplacer now
        ActivationTestClass.standard_sequential_model_valid(model)
        # ActivationTestClass.standard_sequential_model_valid_with_info_printing(model)
        
        # python -m unittest tests.operators.test_activation.ActivationTestClass.test_sequential_flatten_and_backward
        
    @decorator.cuda_test  
    def test_sequential_max_pool2d_and_backward(self):  
        '''Test the max_pool2d() function (fwd+bwd) in a sequential module.'''  
        input = torch.randn((1, 3, 8, 4), dtype=torch.float16, device='cuda')  
        
        #! max_pool2d需要适配特定输入形状。该算子之后可能会抛弃
        class MyModel(torch.nn.Module):
            def __init__(self):
                super(MyModel, self).__init__()
                self.linear = torch.nn.Linear(4, 8, bias=False)
                
            def forward(self, x):
                x = self.linear(x)
                x = Activation.max_pool2d(x, 2, 2)
                return x
        
        model = MyModel().cuda() 
        model1 = copy.deepcopy(model)  
        model1 = LinearReplacer.replace(model1, Dtypes.kfloat16)
        output1 = model1(input)
        # print(f"fp16 model output: {output1}, with dtype: {output1.dtype}")  
          
        # print("------------For fp8 activation model------------")  
        model2 = copy.deepcopy(model)  
        model2 = LinearReplacer.replace(model2, Dtypes.kfloat16, enabling_fp8_activation=True)  
        # print(model2)  
        output2 = model2(input)  
        assert output2.is_fp8_form == True  
        # print(f"output: {output2}, with dtype: {output2.dtype}, requires_grad: {output2.requires_grad}, scaling_meta: {output2.scaling_meta}")  
        output2_fp = TypeCast.cast_from_fp8(output2.view(dtype=torch.uint8), output2.scaling_meta, Dtypes.kfloat16)  
        # print(f"casted output: {output2_fp}")  
          
        # print(f"difference: {output1 - output2_fp}")  
          
        # backward  
        loss1 = Loss_fn.sum(output1)  
        loss1.backward()  
        # print(f"fp16 model weight grad: {model1.linear.weight.grad}, with dtype: {model1.linear.weight.grad.dtype}, with requires_grad: {model1.linear.weight.grad._requires_grad}")  
          
        loss2 = Loss_fn.sum(output2)  
        loss2.backward()  
        # print(f"fp8 model weight grad: {model2.linear.weight.grad}, with dtype: {model2.linear.weight.grad.dtype}, with requires_grad: {model2.linear.weight.grad._requires_grad}")  
          
        assert model2.linear.weight.grad.shape == model1.linear.weight.grad.shape, f"Weight grad shape should be the same as model weight shape {model2.linear.weight.grad.shape}, but got {model1.linear.weight.grad.shape}"
        
        # python -m unittest tests.operators.test_activation.ActivationTestClass.test_sequential_max_pool2d_and_backward
    
    @decorator.cuda_test
    def test_module_layernorm(self):
        '''Test ScalingLayerNorm module.'''
        
        class MyModel(torch.nn.Module):
            def __init__(self):
                super(MyModel, self).__init__()
                self.linear1 = torch.nn.Linear(4, 8)
                self.norm1 = torch.nn.LayerNorm(8)
                self.linear2 = torch.nn.Linear(8, 16)
                self.norm2 = torch.nn.LayerNorm(16)
                self.linear3 = torch.nn.Linear(16, 4)
                
            def forward(self, x):
                x = self.linear1(x)
                x = self.norm1(x)
                x = self.linear2(x)
                x = self.norm2(x)
                x = self.linear3(x)
                return x
            
        model = MyModel().cuda()
        model = ActivationReplacer.replace(model)
        ActivationTestClass.standard_sequential_model_valid(model)
        # ActivationTestClass.standard_sequential_model_valid_with_info_printing(model)
        
        # python -m unittest tests.operators.test_activation.ActivationTestClass.test_module_layernorm
        
    @decorator.cuda_test
    def test_module_dropout(self):
        '''Test ScalingDropout module.'''
        
        class MyModel(torch.nn.Module):
            def __init__(self):
                super(MyModel, self).__init__()
                self.linear1 = torch.nn.Linear(4, 8)
                self.dropout1 = torch.nn.Dropout(0.5)
                self.linear2 = torch.nn.Linear(8, 16)
                self.dropout2 = torch.nn.Dropout(0.5)
                self.linear3 = torch.nn.Linear(16, 4)
                
            def forward(self, x):
                x = self.linear1(x)
                x = self.dropout1(x)
                x = self.linear2(x)
                x = self.dropout2(x)
                x = self.linear3(x)
                return x
            
        model = MyModel().cuda()
        model = ActivationReplacer.replace(model)
        ActivationTestClass.standard_sequential_model_valid(model)
        # ActivationTestClass.standard_sequential_model_valid_with_info_printing(model)
        
        # python -m unittest tests.operators.test_activation.ActivationTestClass.test_module_dropout