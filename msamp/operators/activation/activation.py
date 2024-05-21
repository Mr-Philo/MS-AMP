# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Activation module."""

import torch

from msamp.common.dtype import Dtypes, QType
from msamp.common.utils import Device
from msamp.common.utils import TransformerEngineWrapper as tew
from msamp.common.tensor import ScalingTensor, ScalingMeta
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
            
            #! 按目前的实验来看，te gelu函数不支持uint8类型输入，但是可以产生fp8类型输出
            inp = TypeCast.cast_from_fp8(inp.view(dtype=torch.uint8), inp.scaling_meta, Dtypes.kfloat16)       #! important to(float) method
            
            out_qtype = inp.qtype if out_qtype is None else out_qtype
            
            if Dtypes.is_fp8_qtype(out_qtype):
                
                if not Device.is_fp8_supported():
                    raise RuntimeError("FP8 activation output is not supported on this device.")
                
                out_meta = ScalingMeta(Dtypes.kfloat8_e4m3)
                    
                print(f"before te_gelu: out_meta: {out_meta}, out_qtype: {out_qtype}")
                out = tew.te_gelu(
                    inp,                    # inp
                    out_meta.scale,         # scale
                    out_meta.amax,          # amax_history
                    out_meta.scale_inv,     # scale_inv
                    out_qtype,                      # otype. Here should input msamp.common.dtype.QType, then automatically convert to tex.DType. For example: Dtypes.kfloat16 = QType(name='kFloat16', value=3) --> <DType.kFloat16: 4> 
                )
                print(f"after te_gelu: out_meta: {out_meta}, out_qtype: {out_qtype}")
                
                out = out.view(dtype=torch.float16)
                out.scaling_meta = out_meta
                out.is_fp8_form = True
                return out
            else:
                out = tew.te_gelu(
                    inp,                            # inp
                    cls._empty_tensor,              # scale
                    cls._empty_tensor,              # amax_history
                    cls._empty_tensor,              # scale_inv
                    out_qtype,                      # otype
                )
                out_scaling_tensor = out.cast(Dtypes.kfloat8_e5m2, meta=ScalingMeta(Dtypes.kfloat8_e4m3))   # step1: quantize
                out = out_scaling_tensor.value.view(dtype=torch.float16)                                    # step2: get uint8 tensor, then view to fp16
                out.scaling_meta = out_scaling_tensor.meta                                                  # step3: set scaling meta
                out.is_fp8_form = True                                                                      # step4: set flag
                return out
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
    