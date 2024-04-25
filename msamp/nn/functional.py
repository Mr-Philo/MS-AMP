# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""functional interface in MS-AMP."""

import torch
import torch.nn.functional as F

from msamp.common.dtype import Dtypes
from msamp.common.tensor import ScalingTensor
from msamp.operators.gemm import Gemm
from msamp.nn.state import model_state
from msamp.nn.parameter import ScalingParameter


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
        
        if isinstance(input, torch.Tensor):
            input_meta = metas['input']
            input_fp8 = input.cast(Dtypes.kfloat8_e4m3, meta=input_meta)
            # print(f">>> In _FP8GemmFunction.forward, intime quantization: input: {input_fp8}")    #! temporary
        elif isinstance(input, ScalingTensor):
            input_fp8 = input
            # print(f">>> In _FP8GemmFunction.forward, pre quantization: input: {input_fp8}")       #! temporary
        else:
            raise ValueError(f'input should be either torch.Tensor or ScalingTensor, but got {type(input)}')
        
        weight_fp8 = weight.cast(Dtypes.kfloat8_e4m3)

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
        def new_fn(input, weight, bias=None, enabling_fp8_activation=False):
            r"""linear(input, weight, bias=None, enabling_fp8_activation=False) -> Tensor (or ScalingTensor).

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
            if not enabling_fp8_activation and not isinstance(input, torch.Tensor):
                if not isinstance(input, ScalingTensor):
                    raise TypeError(f'input should be a torch.Tensor. current type: {type(input)}')
                else:
                    raise ValueError('enabling_fp8_activation should be True when input is ScalingTensor.')
            if not isinstance(weight, (torch.Tensor, ScalingTensor)):
                raise TypeError(f'weight should be a torch.Tensor or ScalingTensor. current type: {type(weight)}')
            if bias is not None and not isinstance(bias, torch.Tensor):
                raise TypeError(f'bias should be a torch.Tensor. current type: {type(bias)}')       # todo: how do we check if bias is a ScalingTensor?

            if isinstance(weight, torch.Tensor) and isinstance(input, torch.Tensor) and not hasattr(weight, '_meta'):
                return old_fn(input, weight, bias=bias)

            if not hasattr(weight, '_scaling_metas'):
                raise ValueError('weight (ScalingTensor) should have _scaling_metas attribute.')

            model_state.ready_to_scale_tensor = True
            shape = input.shape

            if len(shape) != 2:
                dim = shape[-1]
                input = input.reshape(-1, dim)

            # todo: output_dtype when enabling_fp8_activation is True
            if isinstance(input, ScalingTensor):
                output_dtype = torch.float16        # todo: very temporary solution
                '''
                Comments on 2024/04/25: if set output_dtype to ScalingTensor.dtypr(torch.uint8), the following error occurs:
                File "/data/ruizhe/MS-AMP/msamp/operators/gemm/gemm.py", line 122, in fp8_gemm
                    tew.te_gemm(
                File "/data/ruizhe/MS-AMP/msamp/common/utils/transformer_engine_wrapper.py", line 103, in te_gemm
                    tex.te_gemm(*new_args)
                RuntimeError: /tmp/pip-install-478citbx/transformer-engine_6eacb10b216149648b925ccd4cfcbcaa/transformer_engine/common/gemm/cublaslt_gemm.cu:267 in function cublas_gemm: Assertion failed: status != CUBLAS_STATUS_NOT_SUPPORTED. Unable to find suitable cuBLAS GEMM algorithm
                '''
            else:
                output_dtype = torch.get_autocast_gpu_dtype() if torch.is_autocast_enabled() else input.dtype
            # print(f">>> In FuctionalOverider._get_wrapper_for_linear, input: {input}")    #! temporary
            
            out = _FP8GemmFunction.apply(input, weight, weight._scaling_metas, cls.EMPTY_GRAD_TENSOR.type(output_dtype))
            if bias is not None:
                out = out + bias.type(output_dtype).view(1, -1)

            if len(shape) != 2:
                out = out.view(shape[:-1] + (-1, ))
                
            if enabling_fp8_activation:
                out = out.cast(Dtypes.kfloat8_e4m3, meta=weight._scaling_metas['ograd'])
                
            return out

        return new_fn


FunctionalOverider.override()
