# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Activation module."""

import torch

from msamp.common.dtype import Dtypes, QType
from msamp.common.utils import Device
from msamp.common.utils import TransformerEngineWrapper as tew
from msamp.common.tensor import ScalingTensor
from msamp.common.tensor import TypeCast


class Activation:
    """Activation class to support FP8 precision"""
    _empty_tensor = torch.Tensor()
    
    @classmethod
    def gelu(
        cls,
        inp: torch.Tensor,
        out_qtype: QType = None,
    )-> torch.Tensor:
        """GeLU activation function with FP8 output"""
        
        if inp.is_fp8_form:
            out_qtype = inp.qtype if out_qtype is None else out_qtype
            
            # out_qtype = Dtypes.kfloat8_e4m3     #! temp
            
            if Device.is_fp8_supported() and False:     #! 目前在真正计算时不考虑FP8计算
                # print(f"scale_inv: {inp.scaling_meta.scale_inv}, type: {type(inp.scaling_meta.scale_inv)}")
                return tew.te_gelu(
                    inp.view(torch.uint8).to(torch.float16),          # inp     #! 似乎并不支持uint8类型输入。所以这么看来的话，也没必要用te的激活函数
                    inp.scaling_meta.scale,         # scale
                    inp.scaling_meta.amax,          # amax_history
                    inp.scaling_meta.scale_inv,     # scale_inv
                    out_qtype,                      # otype. Here should input msamp.common.dtype.QType, then automatically convert to tex.DType. For example: Dtypes.kfloat16 = QType(name='kFloat16', value=3) --> <DType.kFloat16: 4> 
                )
            else:
                inp = TypeCast.cast_from_fp8(inp.view(dtype=torch.uint8), inp.scaling_meta, out_qtype)       #! important to(float) method
                return tew.te_gelu(
                    inp,                            # inp
                    cls._empty_tensor,              # scale
                    cls._empty_tensor,              # amax_history
                    cls._empty_tensor,              # scale_inv
                    out_qtype,                      # otype
                )
                
                #! TODO: enable fp8 activation back
                
        else:
            return torch.nn.functional.gelu(inp)
    
    @classmethod
    def relu(inp: ScalingTensor, out: ScalingTensor) -> ScalingTensor:
        """ReLU activation function with FP8 output"""
        return tew.relu(inp, out)
    
    @classmethod
    def geglu(inp: ScalingTensor, out: ScalingTensor) -> ScalingTensor:
        """GeGLU activation function with FP8 output"""
        return tew.geglu(inp, out)
    
    @classmethod
    def sigmoid(inp: ScalingTensor, out: ScalingTensor) -> ScalingTensor:
        """Sigmoid activation function with FP8 output"""
        return tew.sigmoid(inp, out)
    
    @classmethod
    def tanh(inp: ScalingTensor, out: ScalingTensor) -> ScalingTensor:
        """Tanh activation function with FP8 output"""
        return tew.tanh(inp, out)
    