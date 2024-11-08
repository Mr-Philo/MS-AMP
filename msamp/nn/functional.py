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
import time

USE_W_SIMU_FP4 = bool(int(os.getenv('USE_W_SIMU_FP4', 0)))
USE_A_SIMU_FP4 = bool(int(os.getenv('USE_A_SIMU_FP4', 0)))
USE_W_BACKWARD_SIMU_FP4 = bool(int(os.getenv('USE_W_BACKWARD_SIMU_FP4', USE_W_SIMU_FP4)))      # default same to W forward
USE_A_BACKWARD_SIMU_FP4 = bool(int(os.getenv('USE_A_BACKWARD_SIMU_FP4', USE_A_SIMU_FP4)))      # default same to A forward
USE_G_BACKWARD_SIMU_FP4 = bool(int(os.getenv('USE_G_BACKWARD_SIMU_FP4', 0)))


FP4_QUANTIZATION_TABLE = {
    # 2-D column tensor
    'e0m3': torch.tensor([-6.0, -5.0, -4.0, -3.0, -2.0, -1.0,  0.0, 1.0,  2.0, 3.0, 4.0, 5.0, 6.0]).view(-1, 1).cuda(),
    'e1m2': torch.tensor([-3.0, -2.5, -2.0, -1.5, -1.0, -0.5,  0.0, 0.5,  1.0, 1.5, 2.0, 2.5, 3.0]).view(-1, 1).cuda(),     # eqivalent to e0m3
    'e2m1': torch.tensor([-4.0, -3.0, -2.0, -1.5, -1.0, -0.5,  0.0, 0.5,  1.0, 1.5, 2.0, 3.0, 4.0]).view(-1, 1).cuda(),
    'e3m0': torch.tensor([-8.0, -4.0, -2.0, -1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]).view(-1, 1).cuda(),
}

FP4_QUANTIZATION_TABLE_WIRHOUT_NAN = {
    'e0m3': torch.tensor([-7.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0,  0.0, 1.0,  2.0, 3.0, 4.0, 5.0, 6.0, 7.0]).view(-1, 1).cuda(),
    'e1m2': torch.tensor([-3.5, -3.0, -2.5, -2.0, -1.5, -1.0, -0.5,  0.0, 0.5,  1.0, 1.5, 2.0, 2.5, 3.0, 3.5]).view(-1, 1).cuda(),     # eqivalent to e0m3
    'e2m1': torch.tensor([-6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5,  0.0, 0.5,  1.0, 1.5, 2.0, 3.0, 4.0, 6.0]).view(-1, 1).cuda(),
    'e3m0': torch.tensor([-16,  -8.0, -4.0, -2.0, -1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16 ]).view(-1, 1).cuda(),
}

BASE_4_FP4_QUANTIZATION_TABLE = {
    'e0m3': torch.tensor([-20, -17, -16, -5, -4, -1, 0, 1, 4, 5, 16, 17, 20]).view(-1, 1).cuda(),
    'e1m2': torch.tensor([-5, -4.25, -4, -1.25, -1, -0.25, 0, 0.25, 1, 1.25, 4, 4.25, 5]).view(-1, 1).cuda(),
    'e2m1': torch.tensor([-16, -5, -4, -1.25, -1, -0.25, 0, 0.25, 1, 1.25, 4, 5, 16]).view(-1, 1).cuda(),
    'e3m0': torch.tensor([-64, -16, -4, -1, -0.25, -0.0625, 0, 0.0625, 0.25, 1, 4, 16, 64]).view(-1, 1).cuda(),
}


BASE_4_FP4_QUANTIZATION_TABLE_WIRHOUT_NAN = {
    'e0m3': torch.tensor([-21, -20, -17, -16, -5, -4, -1, 0, 1, 4, 5, 16, 17, 20, 21]).view(-1, 1).cuda(),
    'e1m2': torch.tensor([-5.5, -5, -4.25, -4, -1.25, -1, -0.25, 0, 0.25, 1, 1.25, 4, 4.25, 5, 5.5]).view(-1, 1).cuda(),
    'e2m1': torch.tensor([-20, -16, -5, -4, -1.25, -1, -0.25, 0, 0.25, 1, 1.25, 4, 5, 16, 20]).view(-1, 1).cuda(),
    'e3m0': torch.tensor([-256, -64, -16, -4, -1, -0.25, -0.0625, 0, 0.0625, 0.25, 1, 4, 16, 64, 256]).view(-1, 1).cuda(),
}


def _simu_cast_to_fp4(
    input: torch.Tensor, 
    format: str = 'e1m2', 
    debug_info: bool = False, 
    nan_existed: bool = True, 
    asymmetric: bool = False,
    fp8_test: bool = False, 
    int4_test: bool = False,
    channel_wise: bool = False, 
    token_wise: bool = False,
    use_fp8_sf: bool = False,
    base4: bool = False,
    outlier_clip: bool = False,
    clip_threshold: float = 0.97
) -> torch.Tensor:
    
    """Simulated casting pytorch tensor to fp4. Note: 
    Args:
        input (torch.Tensor): Input tensor to cast.
        format (str): format of fp4, should be in ['e0m3', 'e1m2', 'e2m1', 'e3m0'].
        debug_info (bool): whether to print debug info.
        nan_existed (bool): whether to consider nan in the input tensor. Default is True.
        asymmetric (bool): whether to use asymmetric quantization. Default is False.
        fp8_test (bool): whether to test fp8.
        int4_test (bool): whether to test int4.
        channel_wise (bool): whether to quantize the input tensor channel-wisely.
        token_wise (bool): whether to quantize the input tensor token-wisely. cannot be True at the same time with channel_wise.
        use_fp8_sf (bool): whether to use fp8 scaling factor. If True, the scaling factor will be computed in fp8.
        base4 (bool): whether to use base-4 quantization table. This action will largely expand dynamic range but reduce accuracy. Default is False.
    Return:
        torch.Tensor: whose dtype is still torch.float16 or torch.float32, but numerically quantized into fp4.
    """
    
    # pre-check for formats
    assert isinstance(input, torch.Tensor), f"Input tensor should be torch.Tensor, but got {type(input)}."
    assert format in ['e0m3', 'e1m2', 'e2m1', 'e3m0'], f"Unsupported format: {format}. Please choose from ['e0m3', 'e1m2', 'e2m1', 'e3m0']."

    E, M = (0, 3) if format == 'e0m3' else (1, 2) if format == 'e1m2' else (2, 1) if format == 'e2m1' else (3, 0)
    if int4_test:
        E, M = (0, 3)
    if fp8_test:
        E, M = (4, 3)
        raise NotImplementedError("Currently not supported.")

    # pre-check and reshape for channel-wise or token-wise quantization
    shape = input.shape
    assert not (channel_wise and token_wise), f"channel_wise and token_wise cannot be True at the same time."
    # assert len(input.shape) == 2, f"Input tensor should be 2D, but got {len(input.shape)}D. For channel-wise quantization, please make sure the input activation is in @D shape (batchsize*seq_len, channel dim)."
    if (channel_wise or token_wise) and len(shape) != 2:
        dim = shape[-1]
        input = input.reshape(-1, dim)
    
    # asymmetric quantization    
    if asymmetric:
        zeropoint = (input.min() + input.max()) / 2
        input = input - zeropoint
        
    # outlier clipping
    if outlier_clip:
        time0 = time.time() if debug_info else None
        float_input = input.float()
        if channel_wise:
            input = torch.clamp(input, min=torch.quantile(float_input, 1-clip_threshold, dim=0, keepdim=True), max=torch.quantile(float_input, clip_threshold, dim=0, keepdim=True))
        elif token_wise:
            input = torch.clamp(input, min=torch.quantile(float_input, 1-clip_threshold, dim=1, keepdim=True), max=torch.quantile(float_input, clip_threshold, dim=1, keepdim=True))
        else:
            try:
                input = torch.clamp(input, min=torch.quantile(float_input, 1-clip_threshold), max=torch.quantile(float_input, clip_threshold))
            except RuntimeError:
                # using chunk-wise quantile to deal with large tensor (element number > 16M)
                if debug_info:
                    print(f"Input tensor is too large, using chunk-wise quantile to compute clipping threshold.")
                chunk_size = 1e7    # 10M
                quantiles = []
                for i in range(0, input.numel(), int(chunk_size)):
                    chunk = float_input.view(-1)[i: i + int(chunk_size)]
                    quantiles.append(torch.quantile(chunk, torch.tensor([1-clip_threshold, clip_threshold]).to(chunk.device)))
                quantiles_tensor = torch.stack(quantiles)
                input = torch.clamp(input, min=quantiles_tensor[:, 0].min(), max=quantiles_tensor[:, 1].max())
                if debug_info:
                    print(f"time for clipping {input.numel()} elements: {time.time() - time0}, computed quantiles: {quantiles_tensor[:, 0].min(), quantiles_tensor[:, 1].max()}")

    # get amax
    if channel_wise:
        amax = input.abs().max(dim=0, keepdim=True)[0]      # channel-wise max value
        scale = torch.ones((1, 1), device=input.device)     # 2-D tensor shape
    elif token_wise:
        amax = input.abs().max(dim=1, keepdim=True)[0]      # token-wise max value
        scale = torch.ones((1, 1), device=input.device)     # 2-D tensor shape
    else:
        amax = input.abs().max()
        scale = torch.ones((), device=input.device)
    
    # fp_max = Floating._get_fp_max(E, M, inf_existed=False, nan_existed=nan_existed) if not int4_test else 6.0
    fp_max = 6.0 if E == 0 else 8.0 if M == 0 else Floating._get_fp_max(E, M, inf_existed=False, nan_existed=nan_existed)
    margin = 0
    sf = ScalingMeta.compute_scaling_factor(amax, scale, fp_max, margin)
    if use_fp8_sf:
        sf = sf.cast(Dtypes.kfloat8_e4m3)
        sf = sf.float()     # simulation first
    if debug_info:
        print(f"amax: {amax}, scale: {scale}, fp_max: {fp_max}, margin: {margin}")
        print(f"computed scaling factor: {sf}")

    # result = torch.round(input * sf)        # this * operation can handle matrix-tensor broadcasting. For example, (3, 4) * (4,) -> (3, 4)
    #! look-up table quantization
    def look_up_quantize(x: torch.Tensor, table: torch.Tensor) -> torch.Tensor:
        '''
        'table' tensor should be a 2-D column tensor. if not, using .view(-1, 1) to convert.
        'table' tensor should be on the same device with 'x'. Default to cuda.
        '''
        if len(table.shape) != 2 or table.shape[1] != 1:
            table = table.view(-1, 1)
        if table.device != x.device:
            table = table.to(x.device)
        if table.dtype != x.dtype:
            table = table.to(x.dtype)
        diff_matrix = torch.abs(x.view(1, -1) - table)
        nearest_index = torch.argmin(diff_matrix, dim=0)
        return table[nearest_index].view(x.shape)
    
    # look-up quantization
    if base4:
        TABLE = BASE_4_FP4_QUANTIZATION_TABLE if nan_existed else BASE_4_FP4_QUANTIZATION_TABLE_WIRHOUT_NAN
    else:
        TABLE = FP4_QUANTIZATION_TABLE if nan_existed else FP4_QUANTIZATION_TABLE_WIRHOUT_NAN
    result = look_up_quantize(input * sf, TABLE[format]).div(sf)
    result.requires_grad = input.requires_grad
    
    if debug_info:
        # result = result.to(torch.uint4)       # currently not supported
        print(f"result(in torch.uint-like style): {look_up_quantize(input * sf, TABLE[format])}")
    
    # reshape back for channel-wise or token-wise quantization
    if (channel_wise or token_wise) and len(shape) != 2:
        result = result.view(shape[:-1] + (-1, ))
        
    if asymmetric:
        result = result + zeropoint
        
    return result


# currently not used
def _simu_cast_to_fp4_qualcomm(
    input: torch.Tensor, 
    format: str = 'e2m1', 
    debug_info: bool = False, 
    nan_existed: bool = True, 
) -> torch.Tensor:
    '''
    This method adapted from https://github.com/Qualcomm-AI-research/FP8-quantization/blob/main/quantization/quantizers/fp8_quantizer.py#L91
    '''
    assert isinstance(input, torch.Tensor), f"Input tensor should be torch.Tensor, but got {type(input)}."
    assert format in ['e1m2', 'e2m1'], f"Unsupported format: {format}. Please choose from ['e1m2', 'e2m1']."

    E, M = (1, 2) if format == 'e1m2' else (2, 1)
    fp_max = Floating._get_fp_max(E, M, inf_existed=False, nan_existed=nan_existed)
    bias = 2**E - torch.log2(torch.tensor(fp_max)) + torch.log2(torch.tensor(2 - 2 ** (-M))) - 1
    xc = input
    
    print(torch.floor(torch.log2(torch.abs(xc)) + bias))
    log_scales = torch.clamp((torch.floor(torch.log2(torch.abs(xc)) + bias)).detach(), max = 1.0)
    scales = 2.0 ** (log_scales - M - bias)
    
    if debug_info:
        print(f"fp_max: {fp_max}, bias: {bias}, log_scales: {log_scales}, scales: {scales}")
    
    result = torch.round(xc / scales)
    if debug_info:
        print(f"result(in torch.uint-like style): {result}")
    result = result * scales
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
        
        if not USE_W_SIMU_FP4:
            weight_fp8 = weight.cast(Dtypes.kfloat8_e4m3)
            ctx.weight_fp8 = weight_fp8
        else:
            fp4_weight_in_float = _simu_cast_to_fp4(weight.float(), format='e2m1')
            weight_fp8 = fp4_weight_in_float.cast(Dtypes.kfloat8_e4m3)
            if USE_W_BACKWARD_SIMU_FP4:
                ctx.weight_fp8 = weight_fp8
            else:
                ctx.weight_fp8 = weight.cast(Dtypes.kfloat8_e4m3)
        
        if not USE_A_SIMU_FP4:
            input_fp8 = input.cast(Dtypes.kfloat8_e4m3, meta=input_meta)
            ctx.input_fp8 = input_fp8
        else:
            fp4_input = _simu_cast_to_fp4(input, format='e2m1', channel_wise=True)
            input_fp8 = fp4_input.cast(Dtypes.kfloat8_e4m3, meta=input_meta)
            if USE_A_BACKWARD_SIMU_FP4:
                ctx.input_fp8 = input_fp8
            else:
                ctx.input_fp8 = input.cast(Dtypes.kfloat8_e4m3, meta=input_meta)
        

        ctx.input_fp8.requires_grad = input.requires_grad
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
            fp4_output_grad_in_float = _simu_cast_to_fp4(output_grad, format='e1m2')
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
    
    a = torch.randn(3, 4, dtype=torch.float16).cuda() * 0.01
    # a = torch.randn(1, 1, 1, 3, 4, dtype=torch.float16).cuda() * 0.01
    # a = torch.tensor([[-0.1, 0.2, -0.3], [0.4, -0.5, 0.6], [-0.7, 0.8, -0.9]]).cuda()
    # a = torch.tensor([[-0.01, 0.48, -0.967], [1.623, -2.222, 2.467], [-2.874, 3.3699, -3.457]]).cuda()
    # a = torch.tensor([[0.001, 0.048, 0.0967], [0.1623, 0.2222, 0.2467], [0.2874, 0.33699, 0.3957]]).cuda()      # e2m1 format
    a = torch.tensor([[0.001, 0.048, 0.0997], [0.1503, 0.2002, 0.2497], [0.2974, 0.30699, 0.4001]]).cuda()      # e2m1 format
    # a = torch.tensor([[0.001, 0.048, 0.0967], [0.1623, 0.2222, 0.2467], [0.2467, 0.2874, 0.2998]]).cuda()       # e1m2 format
    a = torch.tensor(
        [ [ [-0.01,  0.48,   -9.67], 
            [1.623,  -2.222, 24.67], ],
          [ [-2.874, 3.699,  -34.57], 
            [0.85,   -1.343, 18.88], ]
        ]
    ).cuda()        # channel-wise outlier. shape: (2, 2, 3)

    # data, meta = simu_cast_to_fp4_using_scaling_meta(a)
    # # b = ScalingTensor(data, meta)     # currently not supported, because te not support kFloat4
    # b = data * meta.scale_inv
    
    if True:       # test channel-wise quantization or token-wise quantization
        # b = _simu_cast_to_fp4(a, format='e2m1', debug_info=True, channel_wise=True)
        b = _simu_cast_to_fp4(a, format='e2m1', debug_info=True, token_wise=True, outlier_clip=True, clip_threshold=0.8)
        # b = _simu_cast_to_fp4(a.permute(0, 2, 1), format='e1m2', debug_info=True, token_wise=True).permute(0, 2, 1)
        c = b.cast(Dtypes.kfloat8_e4m3)

        print(f"Original tensor: {a}, with max value: {a.abs().max()}")
        print(f"Simulated casted tensor to fp4: {b}, with max value: {b.abs().max()}")
        print(f"Double casted tensor to fp8: {c}")
        print(f"ScaingTensor's data: {c.value - 128}")
    
    elif False:   
        b = _simu_cast_to_fp4(a, format='e2m1', debug_info=True)
        # b = _simu_cast_to_fp4(a, format='e2m1', debug_info=True, outlier_clip=True, clip_threshold=0.97)
        # c = a.cast(Dtypes.kfloat8_e4m3)
        d = b.cast(Dtypes.kfloat8_e4m3)
        
        print(f"Original tensor: {a}, with max value: {a.abs().max()}")
        print(f"Simulated casted tensor to fp8: {b}, with max value: {b.abs().max()}")
        # print(f"MS-AMP casted tensor to fp8: {c}")
        # print(f"ScaingTensor's data: {c.value - 128}")
        print(f"Double casted tensor to fp8: {d}")
        
    else:
        aa = (2000*a).cast(Dtypes.kfloat8_e4m3)
        print(aa)
        # fp_max = 4.0
        # margin = 0
        # for amax in range(5, 200, 5):
        #     amax = torch.tensor(0.01 * amax).cuda()
        #     sf = ScalingMeta.compute_scaling_factor(amax, torch.ones((), device='cuda'), fp_max, margin)
        #     print(f"amax: {amax:.2f}, computed scaling factor: {sf}")
