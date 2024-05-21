# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Loss function module."""

import torch

from msamp.common.dtype import Dtypes, QType
from msamp.common.utils import Device
from msamp.common.tensor import ScalingMeta
from msamp.common.tensor import TypeCast


class _ScalingSum(torch.autograd.Function):
    '''Sum function for FP8 input and output.'''
    @staticmethod
    def forward(ctx, inp, dtype=None):
        
        dtype = torch.float16 if dtype is None else dtype
        
        meta = inp.scaling_meta
        ctx.shape = inp.shape             # for example, torch.Size([3,2])
        inp = inp.view(dtype=torch.uint8)
        ctx.true_out_shape = inp.shape    # for example, torch.Size([3,4])
        inp = TypeCast.cast_from_fp8(inp, meta, Dtypes.dtype_to_qtype[dtype])
        
        return inp.sum()
    
    @staticmethod
    def backward(ctx, grad_output):
        grad = grad_output * torch.ones(ctx.true_out_shape, dtype=torch.float16).cuda()  # for example, torch.Size([3,4])
        
        grad_scaling_tensor = grad.cast(Dtypes.kfloat8_e5m2, meta=ScalingMeta(Dtypes.kfloat8_e5m2))    # todo: 在这个SclingSum函数里面，只是为了传回去的activation_grad有scaling_meta这个属性，但是这里这个meta是没有用的，因为毕竟Sum函数的grad_fn是全一。如果之后要实现别的自定义损失函数，这里的meta就有用了，而且可能要考虑forward中的meta值
        grad = grad_scaling_tensor.value.view(dtype=torch.float16)      # torch.Size([3,4]) to torch.Size([3,2])
        grad.scaling_meta = grad_scaling_tensor.meta
        grad.is_fp8_form = True
        
        assert grad.shape == ctx.shape, f"Activation grad shape should be the same as input shape {ctx.shape}, but got {grad.shape}"
        return grad, None


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
            assert inp.dtype == torch.float16, f"Currently _ScalingSum function only supports float16 input tensor when it is in fp8 form, but got {input.dtype}"
            return _ScalingSum.apply(inp, dtype)
        else:
            return torch.sum(inp, dtype)
