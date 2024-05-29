# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Activation module."""

import torch
import warnings

import torch.nn.grad

from msamp.common.dtype import Dtypes, QType
from msamp.common.utils import Device
from msamp.common.utils import TransformerEngineWrapper as tew
from msamp.common.tensor import ScalingTensor, ScalingMeta
from msamp.common.tensor import TypeCast


''' # temporarily not use
class FP8ActivationWrapper(torch.nn.Module):
    def __init__(self, module):
        super(FP8ActivationWrapper, self).__init__()
        self.module = module

    def forward(self, inp_fp8):
        assert inp_fp8.is_fp8_form, "Input must be in FP8 form when using FP8 activation."
        if not inp_fp8.scaling_meta:
            warnings.warn("Input scaling meta is not set. This may cause unexpected behavior. Please check scaling meta is correctly set when using FP8 activation.")
            inp_fp8.scaling_meta = ScalingMeta(Dtypes.kfloat8_e4m3)
        inp_fp16 = TypeCast.cast_from_fp8(inp_fp8.view(dtype=torch.uint8), inp_fp8.scaling_meta, Dtypes.kfloat16)
        out_fp16 = self.module(inp_fp16)
        out_fp8_scaling_tensor = out_fp16.cast(Dtypes.kfloat8_e4m3, meta=ScalingMeta(Dtypes.kfloat8_e4m3))
        out_fp8 = out_fp8_scaling_tensor.value.view(dtype=torch.float16)
        out_fp8.scaling_meta = out_fp8_scaling_tensor.meta
        out_fp8.is_fp8_form = True
        return out_fp8

    def backward(self, grad_output):
        assert grad_output.is_fp8_form, "Grad_output received in backward must be in FP8 form when using FP8 activation."
        if not grad_output.scaling_meta:
            warnings.warn("Grad_output scaling meta is not set. This may cause unexpected behavior. Please check scaling meta is correctly set when using FP8 activation.")
            grad_output.scaling_meta = ScalingMeta(Dtypes.kfloat8_e5m2)
        grad_output_fp16 = TypeCast.cast_from_fp8(grad_output.view(dtype=torch.uint8), grad_output.scaling_meta, Dtypes.kfloat16)
        grad_input = self.module.backward(grad_output_fp16)
        grad_input_fp8_scaling_tensor = grad_input.cast(grad_input, Dtypes.kfloat8_e5m2, ScalingMeta(Dtypes.kfloat8_e5m2))
        grad_input_fp8 = grad_input_fp8_scaling_tensor.value.view(dtype=torch.float16)
        grad_input_fp8.scaling_meta = grad_input_fp8_scaling_tensor.meta
        grad_input_fp8.is_fp8_form = True
        return grad_input_fp8
    
    def __repr__(self):
        return f"FP8ActivationWrapper({self.module})"
'''    
    
class _GeluFunction(torch.autograd.Function):
    '''Gelu function for FP8 input and output.'''
    @staticmethod
    def forward(ctx, inp: torch.Tensor) -> torch.Tensor:
        
        assert inp.is_fp8_form, "This _GeluFunction should only be called with FP8 input. Please check if the input tensor is in FP8 form."
            
        #! 按目前的实验来看，te gelu函数不支持uint8类型输入，但是可以产生fp8类型输出
        inp = TypeCast.cast_from_fp8(inp.view(dtype=torch.uint8), inp.scaling_meta, Dtypes.kfloat16)       #! important to(float) method
        out_qtype = Dtypes.kfloat8_e4m3     # we assume that if input is in fp8 form, then output should be in fp8 form

        if not Device.is_fp8_supported():
            raise RuntimeError("FP8 activation output is not supported on this device.")
        
        out_meta = ScalingMeta(Dtypes.kfloat8_e4m3) 
        # print(f"before te_gelu: out_meta: {out_meta}, out_qtype: {out_qtype}")      #! temp
        out = tew.te_gelu(
            inp,                    # inp
            out_meta.scale,         # scale
            out_meta.amax,          # amax_history
            out_meta.scale_inv,     # scale_inv
            out_qtype,              # otype. Here should input msamp.common.dtype.QType, then automatically convert to tex.DType. For example: Dtypes.kfloat16 = QType(name='kFloat16', value=3) --> <DType.kFloat16: 4> 
        )
        # print(f"after te_gelu: out_meta: {out_meta}, out_qtype: {out_qtype}")       #! temp
        
        out = out.view(dtype=torch.float16)
        out.scaling_meta = out_meta
        out.is_fp8_form = True
        
        ctx.save_for_backward(out)          # save for backward. This function won't save out.scaling_meta
        ctx.out_scaling_meta = out_meta     # manually save out_meta for backward
        
        return out
    
        '''The following code has been proved to have the same effect as the above code
        else:
            out = tew.te_gelu(
                inp,                            # inp
                cls._empty_tensor,              # scale
                cls._empty_tensor,              # amax_history
                cls._empty_tensor,              # scale_inv
                out_qtype,                      # otype
            )
            out_scaling_tensor = out.cast(Dtypes.kfloat8_e4m3, meta=ScalingMeta(Dtypes.kfloat8_e4m3))   # step1: quantize
            out = out_scaling_tensor.value.view(dtype=torch.float16)                                    # step2: get uint8 tensor, then view to fp16
            out.scaling_meta = out_scaling_tensor.meta                                                  # step3: set scaling meta
            out.is_fp8_form = True                                                                      # step4: set flag
            return out
        '''

    @staticmethod
    def backward(ctx, grad_output):
        
        assert grad_output.is_fp8_form, "This _GeluFunction backward should only be called with FP8 grdient. Please check if the gradient back from next layer is in FP8 form."
        grad_output = TypeCast.cast_from_fp8(grad_output.view(dtype=torch.uint8), grad_output.scaling_meta, Dtypes.kfloat16)
        print(f">>> grad_output: {grad_output}")      #! temp
        
        out, = ctx.saved_tensors
        out = TypeCast.cast_from_fp8(out.view(dtype=torch.uint8), ctx.out_scaling_meta, Dtypes.kfloat16)
        print(f">>> backward fp16 out: {out}")      #! temp
        
        input_grad = tew.te_dgelu(grad_output, out, Dtypes.kfloat16)
        print(f">>> input_grad: {input_grad}")      #! temp
        
        input_grad_scaling_tensor = input_grad.cast(Dtypes.kfloat8_e5m2, meta=ScalingMeta(Dtypes.kfloat8_e5m2))
        input_grad = input_grad_scaling_tensor.value.view(dtype=torch.float16)
        input_grad.scaling_meta = input_grad_scaling_tensor.meta
        input_grad.is_fp8_form = True
        return input_grad


class _ReluFunction(torch.autograd.Function):
    '''Relu function for FP8 input and output.'''
    @staticmethod
    def forward(ctx, inp: torch.Tensor) -> torch.Tensor:
        
        assert inp.is_fp8_form, "This _ReluFunction should only be called with FP8 input. Please check if the input tensor is in FP8 form."
        inp = TypeCast.cast_from_fp8_activation(inp)
        
        out = torch.nn.functional.relu(inp)
        out = TypeCast.cast_to_fp8_activation(out, Dtypes.kfloat8_e4m3)
        # ctx.saved_out = out     #! this will cause large memory overhead
        ctx.scaling_meta = out.scaling_meta
        ctx.save_for_backward(out)
        
        return out
    
    @staticmethod
    def backward(ctx, grad_output):
        assert grad_output.is_fp8_form, "This _ReluFunction backward should only be called with FP8 grdient. Please check if the gradient back from next layer is in FP8 form."
        grad_output = TypeCast.cast_from_fp8_activation(grad_output)
        # out = TypeCast.cast_from_fp8_activation(ctx.saved_out)
        out, = ctx.saved_tensors
        out = TypeCast.cast_from_fp8_activation(out, ctx.scaling_meta)
        
        grad_input = grad_output.clone()
        grad_input[out <= 0] = 0
                
        return TypeCast.cast_to_fp8_activation(grad_input, Dtypes.kfloat8_e5m2)
    
    
class _DropoutFunction(torch.autograd.Function):  
    '''Dropout function for FP8 input and output.'''  
      
    @staticmethod  
    def forward(ctx, inp: torch.Tensor, p: float, training: bool = True) -> torch.Tensor:  
        assert inp.is_fp8_form, "This _DropoutFunction should only be called with FP8 input. Please check if the input tensor is in FP8 form."  
        inp = TypeCast.cast_from_fp8_activation(inp)
          
        if training:  
            mask = (torch.rand_like(inp) > p).float() / (1 - p)  
            out = inp * mask  
            ctx.save_for_backward(mask)  
        else:  
            out = inp  
            
        return TypeCast.cast_to_fp8_activation(out, Dtypes.kfloat8_e4m3)
  
    @staticmethod  
    def backward(ctx, grad_output):  
        assert grad_output.is_fp8_form, "This _DropoutFunction backward should only be called with FP8 gradient. Please check if the gradient back from next layer is in FP8 form."  
        grad_output = TypeCast.cast_from_fp8_activation(grad_output)
          
        mask, = ctx.saved_tensors 
          
        # Compute gradient input  
        grad_input = grad_output * mask  
          
        return TypeCast.cast_to_fp8_activation(grad_input, Dtypes.kfloat8_e5m2), None, None
    

class _LogSoftmaxFunction(torch.autograd.Function):  
    '''LogSoftmax function for FP8 input and output.'''  
      
    @staticmethod  
    def forward(ctx, inp: torch.Tensor, dim: int) -> torch.Tensor:  
        # TODO: dim is not a must-have parameter
        assert inp.is_fp8_form, "This _LogSoftmaxFunction should only be called with FP8 input. Please check if the input tensor is in FP8 form."  
        inp = TypeCast.cast_from_fp8_activation(inp)
          
        out = torch.nn.functional.log_softmax(inp, dim=dim)  
        out = TypeCast.cast_to_fp8_activation(out, Dtypes.kfloat8_e4m3)
        # ctx.saved_out = out     #! This will cause large memory overhead
        ctx.scaling_meta = out.scaling_meta
        ctx.save_for_backward(out)
        
        ctx.dim = dim
  
        return out  
  
    @staticmethod  
    def backward(ctx, grad_output):  
        assert grad_output.is_fp8_form, "This _LogSoftmaxFunction backward should only be called with FP8 gradient. Please check if the gradient back from next layer is in FP8 form."  
        grad_output = TypeCast.cast_from_fp8_activation(grad_output)
        # out = TypeCast.cast_from_fp8_activation(ctx.saved_out)
        out, = ctx.saved_tensors
        out = TypeCast.cast_from_fp8_activation(out, ctx.scaling_meta)
            
        grad_input = grad_output - torch.exp(out) * grad_output.sum(dim=ctx.dim, keepdim=True)  
          
        return TypeCast.cast_to_fp8_activation(grad_input, Dtypes.kfloat8_e5m2), None
    
    
class _FlattenFunction(torch.autograd.Function):  
    '''Flatten function for FP8 input and output.'''  
      
    @staticmethod  
    def forward(ctx, inp: torch.Tensor, start_dim: int = 0, end_dim: int = -1) -> torch.Tensor:  
        assert inp.is_fp8_form, "This _FlattenFunction should only be called with FP8 input. Please check if the input tensor is in FP8 form."  
        inp = TypeCast.cast_from_fp8_activation(inp)
        ctx.original_shape = inp.shape
          
        out = torch.flatten(inp, start_dim=start_dim, end_dim=end_dim)
        return TypeCast.cast_to_fp8_activation(out, Dtypes.kfloat8_e4m3)  
  
    @staticmethod  
    def backward(ctx, grad_output):  
        assert grad_output.is_fp8_form, "This _FlattenFunction backward should only be called with FP8 gradient. Please check if the gradient back from next layer is in FP8 form."  
        grad_output = TypeCast.cast_from_fp8_activation(grad_output)
          
        grad_input = grad_output.view(ctx.original_shape)  
          
        return TypeCast.cast_to_fp8_activation(grad_input, Dtypes.kfloat8_e5m2), None, None
    

class _MaxPool2DFunction(torch.autograd.Function):  
    '''MaxPool2D function for FP8 input and output.'''  
      
    @staticmethod  
    def forward(ctx, inp: torch.Tensor, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False) -> torch.Tensor:  
        assert inp.is_fp8_form, "This _MaxPool2DFunction should only be called with FP8 input. Please check if the input tensor is in FP8 form."  
        inp = TypeCast.cast_from_fp8_activation(inp)
          
        # Perform max pooling  
        out, indices = torch.nn.functional.max_pool2d(inp, kernel_size, stride, padding, dilation, ceil_mode, return_indices=True)  
          
        ctx.save_for_backward(inp, indices)  
        ctx.kernel_size = kernel_size  
        ctx.stride = stride  
        ctx.padding = padding  
        ctx.dilation = dilation  
        ctx.ceil_mode = ceil_mode  
          
        return TypeCast.cast_to_fp8_activation(out, Dtypes.kfloat8_e4m3)  
      
    @staticmethod  
    def backward(ctx, grad_output):  
        assert grad_output.is_fp8_form, "This _MaxPool2DFunction backward should only be called with FP8 gradient. Please check if the gradient back from next layer is in FP8 form."  
        grad_output = TypeCast.cast_from_fp8_activation(grad_output)
          
        inp, indices = ctx.saved_tensors  
          
        # Compute gradient of input  
        grad_input = torch.nn.functional.max_unpool2d(grad_output, indices, ctx.kernel_size, ctx.stride, ctx.padding, inp.size())
          
        return TypeCast.cast_to_fp8_activation(grad_input, Dtypes.kfloat8_e5m2), None, None, None, None, None
    
    
''' # temporarily not use
class _Conv2dFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp, conv_module):
        assert inp.is_fp8_form, "This _Conv2dFunction should only be called with FP8 input. Please check if the input tensor is in FP8 form."
        inp = TypeCast.cast_from_fp8_activation(inp)
        
        out = conv_module(inp)
        
        ctx.save_for_backward(inp)
        ctx.conv_module = conv_module
        
        return TypeCast.cast_to_fp8_activation(out, Dtypes.kfloat8_e4m3)

    @staticmethod
    def backward(ctx, grad_output):
        assert grad_output.is_fp8_form, "This _Conv2dFunction backward should only be called with FP8 gradient. Please check if the gradient back from next layer is in FP8 form."
        grad_output = TypeCast.cast_from_fp8_activation(grad_output)
        
        inp, = ctx.saved_tensors
        
        grad_input = torch.nn.grad.conv2d_input(inp.shape, ctx.conv_module.weight, grad_output, stride=ctx.conv_module.stride, padding=ctx.conv_module.padding)
        grad_weight = torch.nn.grad.conv2d_weight(inp, ctx.conv_module.weight.shape, grad_output, stride=ctx.conv_module.stride, padding=ctx.conv_module.padding)
        
        grad_input = TypeCast.cast_to_fp8_activation(grad_input, Dtypes.kfloat8_e5m2) 
        
        grad_weight = grad_weight.cast(Dtypes.kfloat8_e4m3, meta=ScalingMeta(Dtypes.kfloat8_e4m3), sync=True)
        # todo: ctx.weight.backward_grad_update(grad_weight)
        
        return grad_input, grad_weight
'''
    
        
class Activation:
    """Activation class to support FP8 precision"""
    _empty_tensor = torch.Tensor()
    
    @classmethod
    def gelu(cls, inp: torch.Tensor)-> torch.Tensor:
        """GeLU activation function to support FP8 precision"""
        if inp.is_fp8_form:   
            return _GeluFunction.apply(inp)
        else:
            return torch.nn.functional.gelu(inp)
    
    @classmethod
    def relu(cls, inp: torch.Tensor)-> torch.Tensor:
        """ReLU activation function to support FP8 precision"""
        if inp.is_fp8_form:
            return _ReluFunction.apply(inp) 
        else:
            return torch.nn.functional.relu(inp)
    
    @classmethod
    def dropout(cls, inp: torch.Tensor, p: float, training: bool = True)-> torch.Tensor:
        """Dropout function to support FP8 precision"""
        if inp.is_fp8_form:
            return _DropoutFunction.apply(inp, p, training)
        else:
            return torch.nn.functional.dropout(inp, p, training)
        
    @classmethod
    def log_softmax(cls, inp: torch.Tensor, dim: int)-> torch.Tensor:
        """LogSoftmax function to support FP8 precision"""
        if inp.is_fp8_form:
            return _LogSoftmaxFunction.apply(inp, dim)
        else:
            return torch.nn.functional.log_softmax(inp, dim)
        
    @classmethod  
    def flatten(cls, inp: torch.Tensor, start_dim: int = 0, end_dim: int = -1) -> torch.Tensor:  
        """Flatten function to support FP8 precision"""  
        if inp.is_fp8_form:  
            return _FlattenFunction.apply(inp, start_dim, end_dim)  
        else:  
            return torch.flatten(inp, start_dim, end_dim)  
        
    @classmethod
    def max_pool2d(cls, inp: torch.Tensor, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False)-> torch.Tensor:  
        """MaxPool2D function to support FP8 precision"""  
        if inp.is_fp8_form:  
            return _MaxPool2DFunction.apply(inp, kernel_size, stride, padding, dilation, ceil_mode)  
        else:  
            return torch.nn.functional.max_pool2d(inp, kernel_size, stride, padding, dilation, ceil_mode)


''' # temporarily not use
class FP8Conv2DWrapper(torch.nn.Module):
    def __init__(self, module):
        super(FP8Conv2DWrapper, self).__init__()
        self.module = module

    def forward(self, inp):
        if inp.is_fp8_form:
            return _Conv2dFunction.apply(inp, self.module)
        else:
            return self.module(inp)
    
    def __getattr__(self, name):
        if name == "module" or name in self.__dict__:
            return super(FP8Conv2DWrapper, self).__getattr__(name)
        return self.module.__getattribute__(name)

    def __setattr__(self, name, value):
        if name == "module" or name in self.__dict__:
            super(FP8Conv2DWrapper, self).__setattr__(name, value)
        else:
            self.module.__setattr__(name, value)
'''
