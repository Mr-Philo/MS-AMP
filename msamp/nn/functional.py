# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""functional interface in MS-AMP."""

import torch
import torch.nn.functional as F

from msamp.common.dtype import Dtypes
from msamp.common.tensor import ScalingTensor, TypeCast
from msamp.operators.gemm import Gemm
from msamp.nn.state import model_state
from msamp.nn.parameter import ScalingParameter


class _FP8GemmFunction(torch.autograd.Function):
    """A function provides fp8 gemm forward and backward computations."""
    @staticmethod
    def forward(ctx, input, weight, metas, dtype_holder, bias=None, enabling_fp8_activation=False):
        """Forward function.

        Args:
            ctx: Context to store arbitrary data which can be retrieved during the backward pass.
            input (torch.Tensor): Input tensor.
            weight (ScalingTensor): Weight tensor.
            metas (dict): Scaling meta of input, weight and output.
            dtype_holder (torch.Tensor): A tensor to hold the output dtype. The required_grad of this tensor
                should be if input.required_grad is False.
            enabling_fp8_activation (bool): Whether to enable fp8 activation.
        """
        ctx.inp_shape = input.shape
        if len(ctx.inp_shape) != 2:     # deal with input shape other than [batch, feature]
            input = input.reshape(-1, ctx.inp_shape[-1])    # to 2D
                
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
        
        requires_grad = input.requires_grad     # to avoid the cast() or view() operation to change the requires_grad attribute of input tensor
        
        if not enabling_fp8_activation:     # normal forward
            input_meta = metas['input']
            input_fp8 = input.cast(Dtypes.kfloat8_e4m3, meta=input_meta)
        else:
            if not input.is_fp8_form:       # fp16 activation with meta input
                input_meta = metas['input']
                input_fp8 = input.cast(Dtypes.kfloat8_e4m3, meta=input_meta)
            else:                           # fp8 activation with meta input. #! we assume that in this time input is a fp16 tensor viewed from a uint8 tensor
                input_meta = input.scaling_meta
                input = input.view(dtype=torch.uint8)       # fp16 value -> view to uint8 value
                input_fp8 = ScalingTensor(input, meta=input_meta)

        weight_fp8 = weight.cast(Dtypes.kfloat8_e4m3)

        ctx.input_fp8 = input_fp8
        ctx.input_fp8.requires_grad = requires_grad
        ctx.weight_fp8 = weight_fp8
        ctx.weight = weight

        ctx.enabling_fp8_activation = enabling_fp8_activation
        
        if not enabling_fp8_activation:
            output_dtype = dtype_holder.dtype
            output_qtype = Dtypes.dtype_to_qtype[output_dtype]
            
            ctx.output_dtype = output_dtype
            ctx.output_qtype = output_qtype
            
            out = Gemm.fp8_gemm(weight_fp8, input_fp8, output_qtype, use_split_accumulator=False)       # return torch.Tensor
            if bias is not None:
                out = out + bias.type(output_dtype).view(1, -1)
        else:
            output_qtype = Dtypes.kfloat8_e4m3
            ctx.output_dtype = torch.float16
            ctx.output_qtype = output_qtype

            bias = bias.to(torch.float16) if (bias and bias.dtype == torch.float32) else bias
            out = Gemm.fp8_gemm(weight_fp8, input_fp8, output_qtype, bias, use_split_accumulator=False)        # return ScalingTensor
            
            #! assertion 1: bias cannot be in fp32 form
            #! assertion 2: output_qtype should be in the same form as the bias. In other words, if input is fp32, bias is fp16, gemm will fail.

        ctx.gemm_out_shape = out.shape
        
        if len(ctx.inp_shape) != 2:
            out = out.view(ctx.inp_shape[:-1] + (-1, ))      # to original n-D
        ctx.true_out_shape = out.shape
        
        if ctx.enabling_fp8_activation:
            # out = TypeCast.cast_to_fp8_activation(out, Dtypes.kfloat8_e4m3, meta=metas['output'])
            out_meta = out.meta
            out = out.value.view(dtype=torch.float16)
            out.scaling_meta = out_meta
            out.is_fp8_form = True
            return out
        
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
        
        # print(f">>> Starting a new backward pass. received output_grad: {output_grad}, with scaling_meta {output_grad.scaling_meta}")
        
        # some pre-assertion-checks
        if ctx.enabling_fp8_activation:
            assert (output_grad.is_fp8_form and output_grad.scaling_meta is not None), f"output_grad should be a fp8 tensor with scaling_meta, but got outgrad is_fp8_form={output_grad.is_fp8_form} with scaling_meta {output_grad.scaling_meta}"
            assert output_grad.shape[-1]*2 == ctx.true_out_shape[-1]        # view back,见下面分析
            assert output_grad.shape == ctx.true_out_shape[:-1] + (int(ctx.true_out_shape[-1]/2), ), f"activation grad shape should be the same as activation shape {ctx.true_out_shape} (under FP8 activation, the last dimension halfed to {ctx.true_out_shape[:-1] + (ctx.true_out_shape[-1]/2, )}, but got {output_grad.shape}"
        else:
            assert output_grad.shape == ctx.true_out_shape, f"activation grad shape should be the same as activation shape {ctx.true_out_shape} , but got {output_grad.shape}"
            
        # deal with input shape other than [batch, feature]. Note this assertion: len(ctx.out_shape) == len(ctx.inp_shape)
        output_grad_meta = output_grad.scaling_meta     # step1: get meta, see torch.tensor.overrider. avoid the view() operation to destroy the scaling_meta
        if len(ctx.true_out_shape) != 2:
            if ctx.enabling_fp8_activation:
                output_grad = output_grad.view(-1, int(ctx.true_out_shape[-1]/2))    # to 2D
                assert output_grad.shape == ctx.gemm_out_shape[:-1] + (int(ctx.gemm_out_shape[-1]/2), ), f"activation grad shape for gemm computing should be the same as gemm_activation shape {ctx.gemm_out_shape} (under FP8 activation, the last dimension halfed to {ctx.gemm_out_shape[:-1] + (ctx.gemm_out_shape[-1]/2, )}, but got {output_grad.shape}"
            else:
                output_grad = output_grad.view(-1, ctx.true_out_shape[-1])       # to 2D
                assert output_grad.shape == ctx.gemm_out_shape, f"activation grad shape for gemm computing should be the same as gemm_activation shape {ctx.gemm_out_shape}, but got {output_grad.shape}"
            
        # print(f">>>>>> Starting a new backward pass. received output_grad: {output_grad} (after view)")

        # We assign gradients to x.grad directly.
        metas = ctx.metas
        ograd_meta = metas['ograd']
        wgrad_meta = metas['wgrad']
        
        #! TODO: when enabling_fp8_activation is True, how to correctly deal with output_grad?
        if ctx.enabling_fp8_activation:
            output_grad = output_grad.view(dtype=torch.uint8)                   # step2: fp16 view back to uint8: #! shape[-1] doubled
            ograd_fp8 = ScalingTensor(output_grad, meta=output_grad_meta)       # step3: get ScalingTensor
            ograd_fp8_t = ograd_fp8.fp8_transpose()                             # step4: transpose
        else:
            ograd_fp8, ograd_fp8_t = output_grad.fused_cast_transpose(Dtypes.kfloat8_e5m2, meta=ograd_meta)     # fp16 -> fp8 (quantized)

        if ctx.input_fp8.requires_grad:
            weight_fp8_t = ctx.weight_fp8.fp8_transpose()
            
            output_qtype = Dtypes.kfloat8_e5m2 if Dtypes.is_fp8_qtype(ctx.output_qtype) else ctx.output_qtype
            input_grad = Gemm.fp8_gemm(weight_fp8_t, ograd_fp8, output_qtype, use_split_accumulator=True)
            # print(f">>> In _FP8GemmFunction.backward, input_grad for return: {input_grad} (before quant)")    #! temporary
            
            # deal with input shape other than [batch, feature]
            if len(ctx.true_out_shape) != 2:
                input_grad = input_grad.view(ctx.true_out_shape[:-1] + (-1, ))       # to original n-D
            
            if ctx.enabling_fp8_activation:
                #! add activation grad quantization
                # input_grad_scaling_tensor = input_grad.cast(Dtypes.kfloat8_e5m2, meta=metas['agrad'])       # step1: quantize
                # input_grad = input_grad_scaling_tensor.value                                 # step2: get torch.tensor (uint8)
                # input_grad = input_grad.view(dtype=torch.float16)                            # step3: uint8 view to fp16: #!shape[-1] reduced by half
                # input_grad.scaling_meta = input_grad_scaling_tensor.meta                     # step4: set scaling meta, see torch.tensor.overrider
                # input_grad.is_fp8_form = True                                                # step5: set flag
                
                input_grad_meta = input_grad.meta
                input_grad = input_grad.value.view(dtype=torch.float16)
                input_grad.scaling_meta = input_grad_meta
                input_grad.is_fp8_form = True
        else:
            input_grad = None

        # print(f">>> In _FP8GemmFunction.backward, input_grad for return: {input_grad} (after quant), with scaling_meta {input_grad.scaling_meta if input_grad is not None else None}")    #! temporary
        
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
                # todo: how to deal when ctx.output_dtype is FP8e4m3 or FP8e5m2
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
                if not isinstance(wgrad, ScalingTensor):
                    wgrad = wgrad.cast(Dtypes.kfloat8_e4m3, meta=wgrad_meta, sync=True)
                wgrad = wgrad.value.view(-1).view(dtype=torch.float32)
                wgrad.meta = wgrad_meta
                return input_grad, wgrad, None, None, None, None
            elif model_state.use_fp8_ddp:
                # todo: how to deal with this case when wgrad is a ScalingTensor
                wgrad.meta = wgrad_meta
            else:
                # wgrad above this line is torch.Tensor w/o tensor scaling
                if not isinstance(wgrad, ScalingTensor):
                    wgrad = wgrad.cast(Dtypes.kfloat8_e4m3, meta=wgrad_meta, sync=True)

            ctx.weight.backward_grad_update(wgrad)

        # print(f">>> In _FP8GemmFunction.backward, input_grad for return: {input_grad} (before return)")    #! temporary
        return input_grad, None, None, None, None, None


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
            r"""linear(input, weight, bias=None, enabling_fp8_activation=False) -> Tensor.

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
                - Weight (Tensor or (or tuple of (Tensor, ScalingMeta)): :math:`(out\_features, in\_features)` or :math:`(in\_features)`
                - Bias (Tensor or None): :math:`(out\_features)` or :math:`()`
                - Output (Tensor): :math:`(*, out\_features)` or :math:`(*)`, based on the shape of the weight
            """
            if not isinstance(input, torch.Tensor):
                raise TypeError(f'input should be a torch.Tensor. current type: {type(input)}')
            if not isinstance(weight, (torch.Tensor, ScalingTensor)):
                raise TypeError(f'weight should be a torch.Tensor or ScalingTensor. current type: {type(weight)}')
            if bias is not None and not isinstance(bias, torch.Tensor):
                raise TypeError(f'bias should be a torch.Tensor. current type: {type(bias)}')       # todo: how do we check if bias is a ScalingTensor?

            if isinstance(weight, torch.Tensor) and isinstance(input, torch.Tensor) and not hasattr(weight, '_meta'):
                return old_fn(input, weight, bias=bias)

            if not hasattr(weight, '_scaling_metas'):
                raise ValueError('weight (ScalingTensor) should have _scaling_metas attribute.')

            model_state.ready_to_scale_tensor = True

            output_dtype = torch.get_autocast_gpu_dtype() if torch.is_autocast_enabled() else input.dtype    
            out = _FP8GemmFunction.apply(
                input, 
                weight, 
                weight._scaling_metas, 
                cls.EMPTY_GRAD_TENSOR.type(output_dtype), 
                bias,
                enabling_fp8_activation
            )

            return out

        return new_fn


FunctionalOverider.override()
