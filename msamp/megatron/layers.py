# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""The layers module msamp.megatron."""

import torch
from torch.cuda.amp import custom_bwd, custom_fwd

from megatron.core.parallel_state import (
    get_tensor_model_parallel_group, get_tensor_model_parallel_world_size, get_global_memory_buffer
)
from msamp.common.dtype import Dtypes
from msamp.common.tensor import ScalingTensor
from msamp.operators.gemm import Gemm

import os
USE_W_SIMU_FP4 = bool(int(os.getenv('USE_W_SIMU_FP4', 0)))
USE_A_SIMU_FP4 = bool(int(os.getenv('USE_A_SIMU_FP4', 0)))
USE_W_BACKWARD_SIMU_FP4 = bool(int(os.getenv('USE_W_BACKWARD_SIMU_FP4', USE_W_SIMU_FP4)))      # default same to W forward
USE_A_BACKWARD_SIMU_FP4 = bool(int(os.getenv('USE_A_BACKWARD_SIMU_FP4', USE_A_SIMU_FP4)))      # default same to A forward
USE_G_BACKWARD_SIMU_FP4 = bool(int(os.getenv('USE_G_BACKWARD_SIMU_FP4', 0)))

USE_E1M2 = bool(int(os.getenv('USE_E1M2', 0)))
USE_E2M1 = bool(int(os.getenv('USE_E2M1', 0)))
USE_E3M0 = bool(int(os.getenv('USE_E3M0', 0)))

DISABLE_TE_KERNEL = bool(int(os.getenv('DISABLE_TE_KERNEL', 0)))

from msamp.nn.functional import _simu_cast_to_fp4

class FP8LinearWithGradAccumulationAndAsyncCommunication(torch.autograd.Function):
    """A linear function with FP8 support, grad accumulation and async communication."""
    @staticmethod
    @custom_fwd
    def forward(ctx, input, weight, bias, gradient_accumulation_fusion, async_grad_allreduce, sequence_parallel, fp4_quantize_scheme, permit_fp4_computation):
        """Forward pass.

        Args:
            ctx: Context to store arbitrary data which can be retrieved during the backward pass.
            input (torch.Tensor): Input tensor.
            weight (ScalingTensor): Weight tensor.
            bias (torch.Tensor): Bias tensor.
            gradient_accumulation_fusion (bool): Whether to fuse gradient accumulation.
            async_grad_allreduce (bool): Whether to use asynchronous all-reduce.
            sequence_parallel (bool): Whether to use sequence parallel.
            fp4_quantize_scheme (str): Method to control the fp4 activation quantization scheme.
            permit_fp4_computation (bool): Whether to permit simu fp4 gemm function. Note: to actually use fp4, USE_W_SIMU_FP4 or USE_A_SIMU_FP4 must be set to True.

        Returns:
            torch.Tensor: Output tensor.
        """
        
        assert fp4_quantize_scheme in [
            "e2m1_token_wise",
            "e2m1_clip_0.99",
            "e2m1_clip_0.999",      
            "e2m1_mask_clip_0.999",     # for normal activation output + layernorm
            "e2m1_clip_0.9995",
            "e3m0_no_clip",             
            "e3m0_clip_0.999",
            "e2m2_sign_shift",
            "e2m2_sign_shift_clip_0.999",          # for gelu output
            "donot_quantize",           
        ], f"Unsupported fp4_quantize_scheme: {fp4_quantize_scheme}"
        
        a_fp4_args = {
            "e2m1_token_wise":  {"format": "e2m1", "token_wise": True, "outlier_clip": False, "nan_existed": False},
            "e2m1_clip_0.99":   {"format": "e2m1", "token_wise": True, "outlier_clip": True, "clip_threshold": 0.99, "nan_existed": False},
            "e2m1_clip_0.999":  {"format": "e2m1", "token_wise": True, "outlier_clip": True, "clip_threshold": 0.999, "nan_existed": False},
            "e2m1_mask_clip_0.999": {"format": "e2m1", "token_wise": True, "outlier_clip": True, "clip_threshold": 0.999, "nan_existed": False, "use_masked_clip": True},
            "e2m1_clip_0.9995": {"format": "e2m1", "token_wise": True, "outlier_clip": True, "clip_threshold": 0.9995, "nan_existed": False},
            "e3m0_no_clip":     {"format": "e3m0", "token_wise": True, "outlier_clip": False, "nan_existed": False},
            "e3m0_clip_0.999":  {"format": "e3m0", "token_wise": True, "outlier_clip": True, "clip_threshold": 0.999, "nan_existed": False},
            "e2m2_sign_shift":  {"format": "e2m2", "token_wise": True, "outlier_clip": False, "sign_shift": True, "nan_existed": True},
            "e2m2_sign_shift_clip_0.999":  {"format": "e2m2", "token_wise": True, "outlier_clip": True, "clip_threshold": 0.999, "sign_shift": True, "nan_existed": True},
        }
        
        #? this is for fp16 mix fp8 computation.
        if DISABLE_TE_KERNEL:         # use native fp32 computation. refer to megatron-dev/megatron/core/tensor_parallel/layers.py
            ctx.disable_te_kernel = True         #! added. this is for specific weight(ScalingTensor) update in backward pass
            ctx.fp8_weight = weight                 #! added
            
            weight = weight.to(input.dtype)     #! added. see MS-AMP/msamp/common/tensor/tensor.py, Line 160
            
            if permit_fp4_computation and USE_W_SIMU_FP4:
                weight = _simu_cast_to_fp4(weight, format='e2m1')
            if permit_fp4_computation and USE_A_SIMU_FP4:
                input = _simu_cast_to_fp4(input, format='e2m1', channel_wise=True)

            ctx.save_for_backward(input, weight)
            ctx.use_bias = bias is not None
            ctx.gradient_accumulation_fusion = gradient_accumulation_fusion
            ctx.async_grad_allreduce = async_grad_allreduce
            ctx.sequence_parallel = sequence_parallel
            
            if sequence_parallel:
                world_size = get_tensor_model_parallel_world_size()
                dim_size = list(input.size())
                dim_size[0] = dim_size[0] * world_size

                all_gather_buffer = get_global_memory_buffer().get_tensor(dim_size, input.dtype, "mpu")
                torch.distributed._all_gather_base(
                    all_gather_buffer, input, group=get_tensor_model_parallel_group()
                )
                total_input = all_gather_buffer
            else:
                total_input = input

            output = torch.matmul(total_input, weight.t())
            if bias is not None:
                output = output + bias
            return output
        
        # fp8 logic
        ctx.use_bias = bias is not None
        ctx.gradient_accumulation_fusion = gradient_accumulation_fusion
        ctx.async_grad_allreduce = async_grad_allreduce
        ctx.sequence_parallel = sequence_parallel
        ctx.disable_te_kernel = False

        input_shape = input.shape
        ctx.input_shape = input_shape
        metas = weight._scaling_metas
        ctx.metas = metas
        input_meta = metas['input']
        tp_group = get_tensor_model_parallel_group()

        output_dtype = input.dtype
        input = input.contiguous()

        #? this is for fp8 mix simu-fp4 computation.
        if permit_fp4_computation and USE_W_SIMU_FP4:
            if USE_E1M2:
                format_str = 'e1m2'
            elif USE_E2M1:
                format_str = 'e2m1'
            elif USE_E3M0:
                format_str = 'e3m0'
            else:
                format_str = 'e2m1'     # default
            fp4_weight_in_float = _simu_cast_to_fp4(weight.float(), format=format_str, outlier_clip=True, clip_threshold=0.99, nan_existed=False)
            weight_fp8 = fp4_weight_in_float.cast(Dtypes.kfloat8_e4m3)
            if USE_W_BACKWARD_SIMU_FP4:
                ctx.weight_fp8 = weight_fp8
            else:
                ctx.weight_fp8 = weight.cast(Dtypes.kfloat8_e4m3)
        else:
            weight_fp8 = weight.cast(Dtypes.kfloat8_e4m3)
            ctx.weight_fp8 = weight_fp8
            
        old_meta_group = input_meta.group
        input_meta.group = tp_group
        
        if permit_fp4_computation and USE_A_SIMU_FP4 and (fp4_quantize_scheme != 'donot_quantize'):
            fp4_input = _simu_cast_to_fp4(input, **a_fp4_args[fp4_quantize_scheme])
            input_fp8 = fp4_input.cast(Dtypes.kfloat8_e4m3, meta=input_meta, sync=sequence_parallel)
            if USE_A_BACKWARD_SIMU_FP4:
                ctx.input_fp8 = input_fp8
            else:
                ctx.input_fp8 = input.cast(Dtypes.kfloat8_e4m3, meta=input_meta, sync=sequence_parallel)
        else:
            input_fp8 = input.cast(Dtypes.kfloat8_e4m3, meta=input_meta, sync=sequence_parallel)
            ctx.input_fp8 = input_fp8
            
        input_meta.group = old_meta_group

        input_fp8.requires_grad = input.requires_grad
        input = input_fp8.value

        weight_fp8.requires_grad = weight.requires_grad

        # save tensors
        ctx.weight = weight

        dim_size = list(input.size())
        if sequence_parallel:
            assert input.dtype == torch.uint8
            world_size = get_tensor_model_parallel_world_size()
            dim_size[0] = dim_size[0] * world_size

            all_gather_buffer = \
                get_global_memory_buffer().get_tensor(dim_size, input.dtype, 'mpu')
            torch.distributed._all_gather_base(all_gather_buffer, input, group=tp_group)
            total_input = all_gather_buffer
        else:
            total_input = input
        ctx.dim_size = dim_size

        assert total_input.dtype == torch.uint8
        total_input_fp8 = ScalingTensor(total_input.view(-1, input_shape[-1]), input_fp8.meta)

        output_qtype = Dtypes.dtype_to_qtype[output_dtype]
        ctx.output_qtype = output_qtype
        output = Gemm.fp8_gemm(weight_fp8, total_input_fp8, output_qtype, use_split_accumulator=False)
        output = output.view(dim_size[:-1] + [-1])
        if bias is not None:
            output.add_(bias)
        return output

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        """Backward pass.

        Args:
            grad_output (torch.Tensor): Output gradient tensor.

        Returns:
            A tuple of gradients of the arguments.
        """
        
        if ctx.disable_te_kernel:         # temporary disable
            input, weight = ctx.saved_tensors
            use_bias = ctx.use_bias

            if ctx.sequence_parallel:
                world_size = get_tensor_model_parallel_world_size()
                dim_size = list(input.size())
                dim_size[0] = dim_size[0] * world_size

                all_gather_buffer = get_global_memory_buffer().get_tensor(dim_size, input.dtype, "mpu")
                handle = torch.distributed._all_gather_base(
                    all_gather_buffer, input, group=get_tensor_model_parallel_group(), async_op=True
                )

                # Here we rely on CUDA_DEVICE_MAX_CONNECTIONS=1 to ensure that the
                # gather is scheduled before the input gradient computation
                total_input = all_gather_buffer
            else:
                total_input = input
            grad_input = grad_output.matmul(weight)

            if ctx.sequence_parallel:
                handle.wait()

            # Doing gather + slicing during the NeMo forward pass can make this tensor
            # not be contiguous. PyTorch only checks if the tensor is contiguous, and only
            # clones it if it's not contiguous:
            # https://github.com/pytorch/pytorch/blob/c47cf9bc7f9e02f649ab4ed53fe4d35732c92ab6/torch/_refs/__init__.py#L2761
            grad_output = grad_output.contiguous()
            # Convert the tensor shapes to 2D for execution compatibility
            grad_output = grad_output.view(
                grad_output.shape[0] * grad_output.shape[1], grad_output.shape[2]
            )
            total_input = total_input.view(
                total_input.shape[0] * total_input.shape[1], total_input.shape[2]
            )

            if ctx.async_grad_allreduce:
                # Asynchronous all-reduce
                handle = torch.distributed.all_reduce(
                    grad_input, group=get_tensor_model_parallel_group(), async_op=True
                )
                # Here we rely on CUDA_DEVICE_MAX_CONNECTIONS=1 to ensure that the
                # all-reduce is scheduled before the weight gradient computation

            if ctx.sequence_parallel:
                assert not ctx.async_grad_allreduce
                dim_size = list(input.size())
                sub_grad_input = torch.empty(
                    dim_size, dtype=input.dtype, device=torch.cuda.current_device(), requires_grad=False
                )
                # reduce_scatter
                handle = torch.distributed._reduce_scatter_base(
                    sub_grad_input, grad_input, group=get_tensor_model_parallel_group(), async_op=True
                )
                # Here we rely on CUDA_DEVICE_MAX_CONNECTIONS=1 to ensure that the
                # reduce scatter is scheduled before the weight gradient computation

            if ctx.gradient_accumulation_fusion:
                # if weight.main_grad.dtype == torch.float32:
                #     fused_weight_gradient_mlp_cuda.wgrad_gemm_accum_fp32(
                #         total_input, grad_output, weight.main_grad
                #     )
                # elif weight.main_grad.dtype in (torch.float16, torch.bfloat16):
                #     fused_weight_gradient_mlp_cuda.wgrad_gemm_accum_fp16(
                #         total_input, grad_output, weight.main_grad
                #     )
                # else:
                #     raise RuntimeError("Unsupported gradient type for gradient accumulation fusion")
                # grad_weight = None
                raise NotImplementedError("gradient_accumulation_fusion not supported for FP8LinearWithGradAccumulationAndAsyncCommunication")
            else:
                grad_weight = grad_output.t().matmul(total_input)
            grad_bias = grad_output.sum(dim=0) if use_bias else None
            
            #! added: FP8 Weight Gradient
            ctx.fp8_weight.backward_grad_update(grad_weight)
            grad_weight = None

            if ctx.sequence_parallel:
                handle.wait()
                return sub_grad_input, grad_weight, grad_bias, None, None, None, None, None

            if ctx.async_grad_allreduce:
                handle.wait()

            return grad_input, grad_weight, grad_bias, None, None, None, None, None
    
        # fp8 logic
        input_fp8 = ctx.input_fp8
        weight_fp8 = ctx.weight_fp8
        input = input_fp8.value
        output_qtype = ctx.output_qtype
        metas = ctx.metas
        ograd_meta = metas['ograd']

        use_bias = ctx.use_bias

        if ctx.sequence_parallel:
            assert input.dtype == torch.uint8
            world_size = get_tensor_model_parallel_world_size()
            dim_size = list(input.size())
            dim_size[0] = dim_size[0] * world_size

            all_gather_buffer = \
                get_global_memory_buffer().get_tensor(dim_size, input.dtype, 'mpu')
            handle = torch.distributed._all_gather_base(
                all_gather_buffer, input, group=get_tensor_model_parallel_group(), async_op=True
            )

            # Here we rely on CUDA_DEVICE_MAX_CONNECTIONS=1 to ensure that the
            # gather is scheduled before the input gradient computation
            total_input = all_gather_buffer
        else:
            total_input = input

        grad_output = grad_output.contiguous()
            
        input_shape = ctx.input_shape
        output_shape = grad_output.shape
        if len(output_shape) != 2:
            grad_output = grad_output.view(-1, output_shape[-1])
            
        if USE_G_BACKWARD_SIMU_FP4:
            fp4_grad_output_in_float = _simu_cast_to_fp4(grad_output, format='e1m2')
            grad_output_fp8, grad_output_fp8_t = fp4_grad_output_in_float.fused_cast_transpose(Dtypes.kfloat8_e5m2, meta=ograd_meta)
        else:
            grad_output_fp8, grad_output_fp8_t = grad_output.fused_cast_transpose(Dtypes.kfloat8_e5m2, meta=ograd_meta)

        # grad_input
        weight_fp8_t = weight_fp8.fp8_transpose()
        grad_input = Gemm.fp8_gemm(weight_fp8_t, grad_output_fp8, output_qtype, use_split_accumulator=True)
        grad_input = grad_input.view(ctx.dim_size)

        if ctx.sequence_parallel:
            handle.wait()

        total_input_fp8 = ScalingTensor(total_input.view(-1, input_shape[-1]), input_fp8.meta)

        if ctx.async_grad_allreduce:
            # Asynchronous all-reduce
            handle = torch.distributed.all_reduce(grad_input, group=get_tensor_model_parallel_group(), async_op=True)
            # Here we rely on CUDA_DEVICE_MAX_CONNECTIONS=1 to ensure that the
            # all-reduce is scheduled before the weight gradient computation

        if ctx.sequence_parallel:
            assert not ctx.async_grad_allreduce
            dim_size = list(input.size())
            sub_grad_input = torch.empty(
                dim_size, dtype=grad_input.dtype, device=torch.cuda.current_device(), requires_grad=False
            )
            # reduce_scatter
            handle = torch.distributed._reduce_scatter_base(
                sub_grad_input, grad_input, group=get_tensor_model_parallel_group(), async_op=True
            )
            # Here we rely on CUDA_DEVICE_MAX_CONNECTIONS=1 to ensure that the
            # reduce scatter is scheduled before the weight gradient computation

        assert not ctx.gradient_accumulation_fusion, \
            'gradient_accumulation_fusion not supported for FP8LinearWithGradAccumulationAndAsyncCommunication'
        # MS-AMP: compute wgrad
        total_input_fp8_t = total_input_fp8.fp8_transpose()
        wgrad_qtype = output_qtype

        grad_weight = Gemm.fp8_gemm(
            total_input_fp8_t,
            grad_output_fp8_t,
            wgrad_qtype,
            use_split_accumulator=True,
        )

        grad_bias = grad_output.sum(dim=0) if use_bias else None

        # FP8 Weight Gradient
        ctx.weight.backward_grad_update(grad_weight)
        grad_weight = None

        if ctx.sequence_parallel:
            handle.wait()
            return sub_grad_input, grad_weight, grad_bias, None, None, None, None, None

        if ctx.async_grad_allreduce:
            handle.wait()

        return grad_input, grad_weight, grad_bias, None, None, None, None, None
