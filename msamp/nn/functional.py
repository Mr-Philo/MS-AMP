# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""functional interface in MS-AMP."""

import torch
import torch.nn.functional as F

from msamp.common.dtype import Dtypes, Floating
from msamp.common.tensor import ScalingTensor, ScalingMeta
from msamp.operators.gemm import Gemm
from msamp.nn.state import model_state
from msamp.nn.parameter import ScalingParameter

import os

USE_W_SIMU_FP4 = bool(int(os.getenv('USE_W_SIMU_FP4', 0)))
USE_A_SIMU_FP4 = bool(int(os.getenv('USE_A_SIMU_FP4', 0)))
# USE_W_BACKWARD_SIMU_FP4 = bool(int(os.getenv('USE_W_BACKWARD_SIMU_FP4', 0)))      # default same to W forward
# USE_A_BACKWARD_SIMU_FP4 = bool(int(os.getenv('USE_A_BACKWARD_SIMU_FP4', 0)))      # default same to A forward
USE_G_BACKWARD_SIMU_FP4 = bool(int(os.getenv('USE_G_BACKWARD_SIMU_FP4', 0)))


def _simu_cast_to_fp4(input: torch.Tensor, format: str = 'e1m2', debug_info: bool = False, nan_existed: bool = True):
    """Simulated casting pytorch tensor to fp4. Note: 
    Args:
        input (torch.Tensor): Input tensor to cast.
        format (str): format of fp4, should be 'e1m2' or 'e2m1'.
        debug_info (bool): whether to print debug info.
    Return:
        torch.Tensor: whose dtype is still torch.float16 or torch.float32, but numerically quantized into fp4.
    """
    assert isinstance(input, torch.Tensor), f"Input tensor should be torch.Tensor, but got {type(input)}."
    assert format in ['e1m2', 'e2m1'], f"Unsupported format: {format}. Please choose from ['e1m2', 'e2m1']."

    E, M = (1, 2) if format == 'e1m2' else (2, 1)

    amax = input.abs().max()
    scale = torch.ones((), device=input.device)
    fp_max = Floating._get_fp_max(E, M, inf_existed=False, nan_existed=nan_existed)

    margin = 0
    sf = ScalingMeta.compute_scaling_factor(amax, scale, fp_max, margin) * 2
    #! Manually double sf for fp4. This is to adapt to the quantization grid of fp4 (0.5) since the precision of fp4 is too low.
    if debug_info:
        print(f"amax: {amax}, scale: {scale}, fp_max: {fp_max}, margin: {margin}")
        print(f"computed scaling factor: {sf}")

    result = torch.round(input.view(1, -1) * sf).view_as(input)
    if debug_info:
        # result = result.to(torch.uint4)       # currently not supported
        print(f"result(in torch.uint-like style): {result}")

    result.div_(sf)
    result.requires_grad = input.requires_grad
    return result


class _FP8GemmFunction(torch.autograd.Function):
    """A function provides fp8 gemm forward and backward computations."""
    @staticmethod
    def forward(ctx, input, weight, metas, dtype_holder):
        """Forward function.

        Args:
            ctx: Context to store arbitrary data which can be retrieved during the backward pass.
            input (torch.Tensor): Input tensor.
            weight (ScalingTensor): Weight tensor.
            metas (dict): Scaling meta of input, weight and output.
            dtype_holder (torch.Tensor): A tensor to hold the output dtype. The required_grad of this tensor
                should be if input.required_grad is False.
        """
        if isinstance(weight, torch.Tensor) and hasattr(weight, '_meta'):
            padded = weight._padded
            original_shape = weight._original_shape
            meta = weight._meta

            weight = weight.view(dtype=torch.uint8)
            if padded != 0:
                weight = weight[0:weight.numel() - padded]
            weight = weight.view(original_shape)
            weight = ScalingParameter(ScalingTensor(weight, meta))
            ctx.return_wgrad = True

        ctx.metas = metas
        model_state.check_metas_in_flat(metas)
        input_meta = metas['input']
        if USE_W_SIMU_FP4:
            fp4_weight_in_float = _simu_cast_to_fp4(weight.float(), format='e1m2')
            weight_fp8 = fp4_weight_in_float.cast(Dtypes.kfloat8_e4m3)
        else:
            weight_fp8 = weight.cast(Dtypes.kfloat8_e4m3)
        if USE_A_SIMU_FP4:
            input = _simu_cast_to_fp4(input, format='e1m2')
        input_fp8 = input.cast(Dtypes.kfloat8_e4m3, meta=input_meta)
        

        ctx.input_fp8 = input_fp8
        ctx.input_fp8.requires_grad = input.requires_grad
        ctx.weight_fp8 = weight_fp8
        ctx.weight = weight

        output_dtype = dtype_holder.dtype
        output_qtype = Dtypes.dtype_to_qtype[output_dtype]

        ctx.output_dtype = output_dtype
        ctx.output_qtype = output_qtype

        out = Gemm.fp8_gemm(weight_fp8, input_fp8, output_qtype, use_split_accumulator=False)
        return out

    @staticmethod
    def backward(ctx, output_grad):
        """Backward function.

        Args:
            ctx: Context to get the data stored in forward pass.
            output_grad (torch.Tensor): Output gradient tensor.

        Returns:
            tuple (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor): The gradients of the arguments
                in forward function. None if no gradient.
        """
        # pytorch has a bug that output_grad.strides is 0. Use .contiguous() to fix it.
        output_grad = output_grad.contiguous()

        # We assign gradients to x.grad directly.
        metas = ctx.metas
        ograd_meta = metas['ograd']
        wgrad_meta = metas['wgrad']
        if USE_G_BACKWARD_SIMU_FP4:
            fp4_output_grad_in_float = _simu_cast_to_fp4(output_grad, format='e2m1')
            ograd_fp8, ograd_fp8_t = fp4_output_grad_in_float.fused_cast_transpose(Dtypes.kfloat8_e5m2, meta=ograd_meta)
        else:
            ograd_fp8, ograd_fp8_t = output_grad.fused_cast_transpose(Dtypes.kfloat8_e5m2, meta=ograd_meta)

        if ctx.input_fp8.requires_grad:
            weight_fp8_t = ctx.weight_fp8.fp8_transpose()
            input_grad = Gemm.fp8_gemm(weight_fp8_t, ograd_fp8, ctx.output_qtype, use_split_accumulator=True)
        else:
            input_grad = None

        if ctx.weight.requires_grad:
            input_fp8_t = ctx.input_fp8.fp8_transpose()
            wgrad_qtype = ctx.output_qtype
            # compute weight gradient
            if ctx.weight.grad is None:
                wgrad = Gemm.fp8_gemm(
                    input_fp8_t,
                    ograd_fp8_t,
                    wgrad_qtype,
                    use_split_accumulator=True,
                )
            else:
                # gradient accumulation, old_wgrad is FP32 or FP16 without tensor scaling.
                old_wgrad = ctx.weight.grad.to(ctx.output_dtype)
                wgrad = Gemm.fp8_gemm(
                    input_fp8_t,
                    ograd_fp8_t,
                    wgrad_qtype,
                    accumulate=True,
                    out=old_wgrad,
                    use_split_accumulator=True,
                )
                del old_wgrad
            if hasattr(ctx, 'return_wgrad') and ctx.return_wgrad:
                wgrad = wgrad.cast(Dtypes.kfloat8_e4m3, meta=wgrad_meta, sync=True)
                wgrad = wgrad.value.view(-1).view(dtype=torch.float32)
                wgrad.meta = wgrad_meta
                return input_grad, wgrad, None, None
            elif model_state.use_fp8_ddp:
                wgrad.meta = wgrad_meta
            else:
                # wgrad above this line is torch.Tensor w/o tensor scaling
                wgrad = wgrad.cast(Dtypes.kfloat8_e4m3, meta=wgrad_meta, sync=True)

            ctx.weight.backward_grad_update(wgrad)

        return input_grad, None, None, None


class FunctionalOverider:
    """Class to override functions in torch.nn.functional."""
    EMPTY_GRAD_TENSOR = torch.nn.Parameter(torch.tensor([]))

    @classmethod
    def override(cls):
        """Override functions in torch.nn.functional."""
        F.linear = cls._get_wrapper_for_linear(F.linear)

    @classmethod
    def _get_wrapper_for_linear(cls, old_fn):
        """Get wrapper for torch.nn.functional.linear (F.linear)."""
        def new_fn(input, weight, bias=None):
            r"""linear(input, weight, bias=None) -> Tensor.

            Applies a linear transformation to the incoming data: :math:`y = xA^T + b`.

            This operation supports 2-D :attr:`weight` with :ref:`sparse layout<sparse-docs>`

            .. warning::
                Sparse support is a beta feature and some layout(s)/dtype/device combinations may not be supported,
                or may not have autograd support. If you notice missing functionality please
                open a feature request.

            This operator supports :ref:`TensorFloat32<tf32_on_ampere>`.

            Shape:

                - Input (Tensor): :math:`(*, in\_features)` where `*` means any number of
                  additional dimensions, including none
                - Weight (Tensor or ScalingTensor): :math:`(out\_features, in\_features)` or :math:`(in\_features)`
                - Bias (Tensor or None): :math:`(out\_features)` or :math:`()`
                - Output (Tensor): :math:`(*, out\_features)` or :math:`(*)`, based on the shape of the weight
            """
            if not isinstance(input, torch.Tensor):
                raise TypeError(f'input should be a torch.Tensor. current type: {type(input)}')
            if not isinstance(weight, (torch.Tensor, ScalingTensor)):
                raise TypeError(f'weight should be a torch.Tensor or ScalingTensor. current type: {type(weight)}')
            if bias is not None and not isinstance(bias, torch.Tensor):
                raise TypeError(f'bias should be a torch.Tensor. current type: {type(bias)}')

            if isinstance(weight, torch.Tensor) and not hasattr(weight, '_meta'):
                return old_fn(input, weight, bias=bias)

            if not hasattr(weight, '_scaling_metas'):
                raise ValueError('weight (ScalingTensor) should have _scaling_metas attribute.')

            model_state.ready_to_scale_tensor = True
            shape = input.shape

            if len(shape) != 2:
                dim = shape[-1]
                input = input.reshape(-1, dim)

            output_dtype = torch.get_autocast_gpu_dtype() if torch.is_autocast_enabled() else input.dtype
            out = _FP8GemmFunction.apply(input, weight, weight._scaling_metas, cls.EMPTY_GRAD_TENSOR.type(output_dtype))
            if bias is not None:
                out = out + bias.type(output_dtype).view(1, -1)

            if len(shape) != 2:
                out = out.view(shape[:-1] + (-1, ))
            return out

        return new_fn


FunctionalOverider.override()


if __name__ == '__main__':
    
    a = torch.randn(3, 4).cuda() * 0.01
    # a = torch.tensor([[-0.1, 0.2, -0.3], [0.4, -0.5, 0.6], [-0.7, 0.8, -0.9]]).cuda()
    # a = torch.tensor([[-0.01, 0.48, -0.967], [1.623, -2.222, 2.467], [-2.874, 3.3699, -3.457]]).cuda()

    # data, meta = simu_cast_to_fp4_using_scaling_meta(a)
    # # b = ScalingTensor(data, meta)     # currently not supported, because te not support kFloat4
    # b = data * meta.scale_inv

    b = _simu_cast_to_fp4(a, format='e1m2', debug_info=True)
    c = b.cast(Dtypes.kfloat8_e4m3)

    print(f"Original tensor: {a}")
    print(f"Simulated casted tensor to fp4: {b}")
    print(f"Double casted tensor to fp8: {c}")
