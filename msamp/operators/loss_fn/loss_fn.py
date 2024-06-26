# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Loss function module."""

import torch

from msamp.common.dtype import Dtypes
from msamp.common.tensor import TypeCast


class _ScalingSum(torch.autograd.Function):
    '''Sum function for FP8 input and output.'''
    @staticmethod
    def forward(ctx, inp, dtype=None):
        
        ctx.dtype = inp.dtype if dtype is None else dtype
        
        ctx.shape = inp.shape
        ctx.true_out_shape = inp.shape[:-1] + (inp.shape[-1]*2 if ctx.dtype == torch.float16 else inp.shape[-1]*4, )    # select between torch.float16 and torch.float32 

        inp = TypeCast.cast_from_fp8_activation(inp)
        
        return inp.sum()
    
    @staticmethod
    def backward(ctx, grad_output):
        grad = grad_output * torch.ones(ctx.true_out_shape, dtype=ctx.dtype).cuda()
        
        return TypeCast.cast_to_fp8_activation(grad, Dtypes.kfloat8_e5m2), None


class _ScalingNLLLoss(torch.autograd.Function):  
    '''NLLLoss function for FP8 input and output.'''  
      
    @staticmethod  
    def forward(ctx, inp, target):  
        ctx.shape = inp.shape  
        ctx.dtype = inp.dtype
        ctx.true_out_shape = inp.shape[:-1] + (inp.shape[-1]*2 if ctx.dtype == torch.float16 else inp.shape[-1]*4, )       # same as inp.view(dtype=torch.uint8).shape
        inp = TypeCast.cast_from_fp8_activation(inp)
        ctx.save_for_backward(inp, target)  

        return torch.nn.functional.nll_loss(inp, target)    
      
    @staticmethod  
    def backward(ctx, grad_output):  
        inp, target = ctx.saved_tensors  
          
        # Compute the gradient of the loss w.r.t the input  
        N = inp.size(0)
        grad_input = torch.zeros(ctx.true_out_shape, dtype=ctx.dtype).cuda()
        # Calculate the gradient of the loss with respect to input
        grad_input.scatter_(1, target.unsqueeze(1), -1.0 / N)
        # Scale by the incoming gradient (chain rule)
        grad_input *= grad_output  
          
        return TypeCast.cast_to_fp8_activation(grad_input, Dtypes.kfloat8_e5m2), None

class Loss_fn:
    """Loss function class to FP8 precision"""
    
    @classmethod
    def sum(cls, inp: torch.Tensor, dtype=None) -> torch.Tensor:
        """Sum function for FP8 input and output.

        Args:
            inp (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        if inp.is_fp8_form:
            assert inp.dtype in [torch.float16, torch.float32], f"Currently _ScalingSum function only supports float16 or float32 input tensor when it is in fp8 form, but got {inp.dtype}"
            return _ScalingSum.apply(inp, dtype)
        else:
            return torch.sum(inp, dtype)
        
    @classmethod
    def nll_loss(cls, inp: torch.Tensor, target: torch.Tensor) -> torch.Tensor:  
        """NLLLoss function for FP8 input and output.  
          
        Args:  
            inp (torch.Tensor): Input tensor.  
            target (torch.Tensor): Target tensor.  
          
        Returns:  
            torch.Tensor: Output tensor.  
        """  
        if inp.is_fp8_form:  
            assert inp.dtype in [torch.float16, torch.float32], f"Currently _ScalingNLLLoss function only supports float16 or float32 input tensor when it is in fp8 form, but got {inp.dtype}" 
            return _ScalingNLLLoss.apply(inp, target)  
        else:  
            return torch.nn.functional.nll_loss(inp, target)
