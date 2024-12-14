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
import numpy as np
from typing import Literal
from sklearn.ensemble import IsolationForest


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
    'e2m2': torch.tensor([0.0,  0.25,  0.5, 0.75,  1.0, 1.25,  1.5, 1.75, 2.0,  2.5, 3.0, 3.5, 4.0, 5.0, 6.0]).view(-1, 1).cuda(),     # only positive
}

FP4_QUANTIZATION_TABLE_WIRHOUT_NAN = {
    'e0m3': torch.tensor([-7.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0,  0.0, 1.0,  2.0, 3.0, 4.0, 5.0, 6.0, 7.0]).view(-1, 1).cuda(),
    'e1m2': torch.tensor([-3.5, -3.0, -2.5, -2.0, -1.5, -1.0, -0.5,  0.0, 0.5,  1.0, 1.5, 2.0, 2.5, 3.0, 3.5]).view(-1, 1).cuda(),     # eqivalent to e0m3
    'e2m1': torch.tensor([-6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5,  0.0, 0.5,  1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.bfloat16).view(-1, 1).cuda(),
    'e3m0': torch.tensor([-16,  -8.0, -4.0, -2.0, -1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16 ]).view(-1, 1).cuda(),
    'e2m2': torch.tensor([0.0,  0.25,  0.5, 0.75,  1.0, 1.25,  1.5, 1.75, 2.0,  2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 7.0]).view(-1, 1).cuda(),     # only positive
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


def apply_iForest_outlier_clipping(input, use_mask=False, debug_info=False, return_residual=False):
    time0 = time.time() if debug_info else None
    
    # 使用孤立森林检测异常值
    clf = IsolationForest(
        contamination=0.001, 
        n_jobs=-1,              # use all available CPUs
        random_state=42         # for reproducibility
    )
    input_reshaped = input.float().flatten().cpu().numpy().reshape(-1, 1)
    outlier_labels = clf.fit_predict(input_reshaped)
    
    print(f"[iForest] detected {sum(outlier_labels == -1)} outliers in {input.numel()} elements.")
    #  if debug_info else None
    outlier_indices = np.where(outlier_labels == -1)[0]
    weights_filtered = np.delete(input_reshaped, outlier_indices, axis=0)
    
    lower_bound = weights_filtered.min()
    upper_bound = weights_filtered.max()
    if use_mask:
        output = torch.where((input >= lower_bound) & (input <= upper_bound), input, torch.tensor(0.0, device=input.device))
    else:
        output = torch.clamp(input, min=lower_bound, max=upper_bound)
    print(f"[iForest] clipping range: {lower_bound, upper_bound}") if debug_info else None
    print(f"[iForest] time for clipping {input.numel()} elements: {time.time() - time0}") if debug_info else None
    
    if return_residual:
        return output, input - output
    else:
        return output, None
    
    
def apply_quantile_clipping(input, clip_threshold, channel_wise=False, token_wise=False, use_mask=False, debug_info=False, return_residual=False):
    time0 = time.time() if debug_info else None
    float_input = input.float() if input.dtype != torch.float32 else input
    if channel_wise:
        # 计算每个通道的上限和下限分位数
        lower_bound = torch.quantile(float_input, 1 - clip_threshold, dim=0, keepdim=True)
        upper_bound = torch.quantile(float_input, clip_threshold, dim=0, keepdim=True)
        if use_mask:
            # 使用掩码，将超出范围的元素置为0
            output = torch.where((input >= lower_bound) & (input <= upper_bound), input, torch.tensor(0.0, device=input.device))
        else:
            # 使用 clamp 裁剪
            output = torch.clamp(input, min=lower_bound, max=upper_bound)
    
    elif token_wise:
        # 计算每个 token 的上限和下限分位数
        lower_bound = torch.quantile(float_input, 1 - clip_threshold, dim=1, keepdim=True)
        upper_bound = torch.quantile(float_input, clip_threshold, dim=1, keepdim=True)
        if use_mask:
            output = torch.where((input >= lower_bound) & (input <= upper_bound), input, torch.tensor(0.0, device=input.device))
        else:
            output = torch.clamp(input, min=lower_bound, max=upper_bound)

    else:
        try:
            # 计算整个张量的上限和下限分位数
            lower_bound = torch.quantile(float_input, 1 - clip_threshold)
            upper_bound = torch.quantile(float_input, clip_threshold)
        except RuntimeError:
            # 使用分块计算分位数以处理大张量
            if debug_info:
                print(f"Input tensor is too large, using chunk-wise quantile to compute clipping threshold.")
            chunk_size = 1e7  # 10M
            quantiles = []
            for i in range(0, input.numel(), int(chunk_size)):
                chunk = float_input.view(-1)[i: i + int(chunk_size)]
                quantiles.append(torch.quantile(chunk, torch.tensor([1 - clip_threshold, clip_threshold]).to(chunk.device)))
            quantiles_tensor = torch.stack(quantiles)
            lower_bound = quantiles_tensor[:, 0].min()
            upper_bound = quantiles_tensor[:, 1].max()
        if use_mask:
            output = torch.where((input >= lower_bound) & (input <= upper_bound), input, torch.tensor(0.0, device=input.device))
        else:
            output = torch.clamp(input, min=lower_bound, max=upper_bound)
    if debug_info:
        print(f"time for clipping {input.numel()} elements: {time.time() - time0}, computed quantiles: {lower_bound, upper_bound}")

    if return_residual:
        return output, input - output
    else:
        return output, None


def _simu_cast_to_fp4(
    input: torch.Tensor, 
    format: Literal['e0m3', 'e1m2', 'e2m1', 'e3m0', 'e2m2'] = 'e2m1',
    quantize_method: Literal['rtn', 'stochastic'] = 'rtn',
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
    clip_method: Literal['iForest', 'quantile'] = 'quantile',
    clip_threshold: float = 0.97,
    use_masked_clip: bool = False,
    apply_vector_clip: bool = True,
    sign_shift: bool = False,
    residual_compensation: bool = False,
    return_residual: bool = False,
    direct_return_after_clip: bool = False,
    return_scaled_input_for_bwd: bool = False,
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
    assert format in ['e0m3', 'e1m2', 'e2m1', 'e3m0', 'e2m2'], f"Unsupported format: {format}. Please choose from ['e0m3', 'e1m2', 'e2m1', 'e3m0', 'e2m2']."

    E, M = (0, 3) if format == 'e0m3' else (1, 2) if format == 'e1m2' else (2, 1) if format == 'e2m1' else (3, 0) if format == 'e3m0' else (2, 2)
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
        
    # sign shift. currently only support positive
    if sign_shift:
        shift = input.min()
        input = input - shift         # shift to positive, and min value to zero
        
    # outlier clipping
    if outlier_clip:
        if clip_method == 'iForest':
            input, residual = apply_iForest_outlier_clipping(input, use_mask=use_masked_clip, debug_info=debug_info, return_residual=(return_residual or residual_compensation))
        elif clip_method == 'quantile':
            # use quantile clipping
            if apply_vector_clip:
                channel_wise_clip, token_wise_clip = channel_wise, token_wise
            else:
                channel_wise_clip, token_wise_clip = False, False
            input, residual = apply_quantile_clipping(input, clip_threshold, channel_wise_clip, token_wise_clip, use_masked_clip, debug_info, return_residual=(return_residual or residual_compensation))
    
    if direct_return_after_clip:
        if (channel_wise or token_wise) and len(shape) != 2:
            result = result.view(shape[:-1] + (-1, ))
        if sign_shift:
            result = result + shift
        return input

    # asymmetric quantization    
    if asymmetric:
        zeropoint = (input.min() + input.max()) / 2
        input = input - zeropoint
        
        
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
    
    # handle fp_max when E=0 or M=0
    if E == 0:
        fp_max = 6.0 if nan_existed else 7.0
    elif M == 0:
        fp_max = 8.0 if nan_existed else 16.0
    else:
        fp_max = Floating._get_fp_max(E, M, inf_existed=False, nan_existed=nan_existed)
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
            
        if True:       #! matrix operation    O(M*N), M: len(input), N: len(table)
            diff_matrix = torch.abs(x.view(1, -1) - table)
            nearest_index = torch.argmin(diff_matrix, dim=0)
            return table[nearest_index].view(x.shape)
        else:           #! binary search       O(M*logN), M: len(input), N: len(table)
            table = table.view(-1)      # TODO: 确定使用随机量化后，原table直接使用一维tensor即可
            indices = torch.searchsorted(table, x, right=True)
            indices = torch.clamp(indices, 1, len(table) - 1)  # Ensure valid range
            left_values = table[indices - 1]
            right_values = table[indices]
            left_closer = (torch.abs(x - left_values) <= torch.abs(x - right_values))
            nearest_values = torch.where(left_closer, left_values, right_values)
            return nearest_values.view(x.shape)
    
    def stochastic_quantize(x: torch.Tensor, table: torch.Tensor) -> torch.Tensor:
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
        
        table = table.view(-1)      # TODO: 确定使用随机量化后，原table直接使用一维tensor即可
        # 二分查找，返回右侧的索引 / Binary search: compute nearest indices for each element in x
        indices = torch.searchsorted(table, x, right=True)
        
        # Clamp indices to valid range
        indices = torch.clamp(indices, 1, len(table) - 1) 
        left_values = table[indices - 1]
        right_values = table[indices]
        
        # Compute distances to the two nearest points
        left_diff = torch.abs(x - left_values)
        right_diff = torch.abs(x - right_values)
        total_diff = left_diff + right_diff
        
        # Compute probabilities for stochastic rounding
        probs = right_diff / total_diff  # Probability of choosing left
        random_vals = torch.rand_like(probs)  # Generate random numbers in [0, 1]
        
        # Stochastic rounding decision
        quantized_values = torch.where(random_vals < probs, left_values, right_values)
        
        return quantized_values.view(x.shape)
    
    # look-up quantization
    if base4:
        raise NotImplementedError("Do not use base4 quantization")
        TABLE = BASE_4_FP4_QUANTIZATION_TABLE if nan_existed else BASE_4_FP4_QUANTIZATION_TABLE_WIRHOUT_NAN
    else:
        TABLE = FP4_QUANTIZATION_TABLE if nan_existed else FP4_QUANTIZATION_TABLE_WIRHOUT_NAN
        
    quantize_func = look_up_quantize if quantize_method == 'rtn' else stochastic_quantize
    scaled_input = input * sf       # this * operation can handle matrix-tensor broadcasting. For example, (3, 4) * (4,) -> (3, 4)
    result = quantize_func(scaled_input, TABLE[format]).div(sf)        # this .div() method can also handle matrix-tensor broadcasting
    if residual_compensation:
        result = result + residual
    result.requires_grad = input.requires_grad
    
    if debug_info:
        # result = result.to(torch.uint4)       # currently not supported
        print(f"result(in torch.uint-like style): {quantize_func(scaled_input, TABLE[format])}")
        # 统计每个数出现了多少次
        print(f"unique values and their counts: {torch.unique(quantize_func(scaled_input, TABLE[format]).float(), return_counts=True)}")
    
    # reshape back for channel-wise or token-wise quantization
    if (channel_wise or token_wise) and len(shape) != 2:
        result = result.view(shape[:-1] + (-1, ))
        if return_residual:
            residual = residual.view(shape[:-1] + (-1, ))
        if return_scaled_input_for_bwd:
            scaled_input = scaled_input.view(shape[:-1] + (-1, ))
        
    if asymmetric:
        result = result + zeropoint
    if sign_shift:
        result = result + shift
       
    assert not (return_scaled_input_for_bwd and return_residual), f"return_scaled_input_for_bwd and return_residual cannot be True at the same time." 
    if return_residual:
        return result, residual
    elif return_scaled_input_for_bwd:
        return result, scaled_input
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


@torch.no_grad()
def _differentiable_quantize_derivative(x: torch.Tensor, k=5, level_format='e2m1', nan_existed=False, using_scaled_k=True, use_tanh=False, power_clamp_max=3.0):
    
    def _tanh_differentiable_quantize_derivative_single_step(x, delta, k):
        sech_squred = 1 / torch.cosh(k * (x - delta / 2))**2
        return (delta * k * sech_squred) / (2 * torch.tanh(k * delta / 2))
    
    def _power_differentiable_quantize_derivative_single_step(x, delta, k):
        return torch.clamp(torch.abs(2 * x / delta - 1) ** (1 / k - 1) / k, max=power_clamp_max) 

    levels = FP4_QUANTIZATION_TABLE[level_format].flatten() if nan_existed else FP4_QUANTIZATION_TABLE_WIRHOUT_NAN[level_format].flatten()
    levels = levels.to(x.device)
    levels = levels.to(x.dtype)
    dy = torch.zeros_like(x)
    for i in range(len(levels) - 1):
        # Determine the interval and delta
        left = levels[i]
        right = levels[i + 1]
        delta = right - left
        # Apply smooth quantization for this interval
        mask = (x >= left) & (x < right)  # Mask for this interval
        if use_tanh:
            if using_scaled_k:
                k_d = k / delta     #! Scaled k by delta
            else:
                k_d = k
            dy[mask] = _tanh_differentiable_quantize_derivative_single_step(x[mask] - left, delta, k_d)
        else:
            dy[mask] = _power_differentiable_quantize_derivative_single_step(x[mask] - left, delta, k)
    return dy


@torch.no_grad()
def _advanced_differentiable_quantize_derivative(x, k=25, level_format='e2m1', nan_existed=False, ste_smooth_alpha=0.0):
    '''
    update 1203
    for speed up, we use matrix operation to compute the derivative for all intervals.
    and we add a control parameter 'ste_smooth_alpha' to control the smoothness of the derivative.
    '''
    
    levels = FP4_QUANTIZATION_TABLE[level_format].flatten() if nan_existed else FP4_QUANTIZATION_TABLE_WIRHOUT_NAN[level_format].flatten()
    levels = levels.to(x.device).to(x.dtype)
    
    # Compute delta for all intervals
    deltas = levels[1:] - levels[:-1]
    
    # Broadcast x to all intervals
    x_expanded = x.unsqueeze(-1)    # Shape: (n, 1) if x is 1D tensor
    levels_left = levels[:-1].unsqueeze(0)  # Shape: (1, m-1)
    levels_right = levels[1:].unsqueeze(0)  # Shape: (1, m-1)
    
    # Mask for all intervals
    mask = (x_expanded >= levels_left) & (x_expanded < levels_right)  # Shape: (n, m-1)
    
    # compute differentiable quantize derivative for all intervals
    delta_expanded = deltas.unsqueeze(0)  # Shape: (1, m-1)
    shifted_x = x_expanded - levels_left  # Shape: (n, m-1), shifting x to [0, delta]
    
    sech_squred = 1 / torch.cosh(k * (shifted_x - delta_expanded / 2))**2
    # derivatives = (1 - ste_smooth_alpha) * (delta_expanded * k * sech_squred) / (2 * torch.tanh(k * delta_expanded / 2)) + ste_smooth_alpha * torch.ones_like(shifted_x)
    derivatives = (delta_expanded * k * sech_squred) / (2 * torch.tanh(k * delta_expanded / 2))
    
    # Combine results using the mask
    dy = torch.sum(derivatives * mask, dim=-1)  # Summing over all intervals
    return dy
    

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


def test_sparsified_compensation():
    
    original_tensor = torch.randn(1000, 2000, dtype=torch.bfloat16).cuda() * 0.01
    fp4_tensor, residual = _simu_cast_to_fp4(original_tensor, format='e2m1', debug_info=True, residual_compensation=False, outlier_clip=True, clip_threshold=0.8, return_residual=True)
    # residual should be sparse
    print(f"Residual sparsity: {residual.to_sparse()._values().numel() / residual.numel()}")
    compensated_tensor = fp4_tensor + residual
    
    # before compensation
    cos_similarity = torch.nn.functional.cosine_similarity(original_tensor.view(-1), fp4_tensor.view(-1), dim=0)
    snr = 10 * torch.log10((original_tensor.norm() / (original_tensor - fp4_tensor).norm())**2)
    print(f"Before compensation, cosine similarity: {cos_similarity}, SNR: {snr}")
    # after compensation
    cos_similarity = torch.nn.functional.cosine_similarity(original_tensor.view(-1), compensated_tensor.view(-1), dim=0)
    snr = 10 * torch.log10((original_tensor.norm() / (original_tensor - compensated_tensor).norm())**2)
    print(f"After compensation, cosine similarity: {cos_similarity}, SNR: {snr}")
    
    # valid matrix multiplication
    k = 50000
    weight = torch.randn(2000, 1000, dtype=torch.bfloat16).cuda() * 0.01
    start_time = time.time()
    for i in range(k):
        output = torch.matmul(original_tensor, weight)
    end_time = time.time()
    print(f"Average time for original matrix multiplication: {(end_time - start_time) / k}")
    
    sparse_residual = residual.to_sparse()
    start_time = time.time()
    #! sparse matrix multiplication must be in fp32
    sparse_residual = sparse_residual.to(torch.float32)
    weight = weight.to(torch.float32)
    for i in range(k):
        res_output = torch.matmul(sparse_residual, weight)
    end_time = time.time()
    print(f"Average time for sparse matrix multiplication: {(end_time - start_time) / k}")
    
    true_out = torch.matmul(fp4_tensor, weight.to(fp4_tensor.dtype)) + res_output.to(residual.dtype)
    # true_out = torch.matmul(fp4_tensor, weight) + torch.matmul(residual, weight)
    compensated_out = torch.matmul(compensated_tensor, weight.to(fp4_tensor.dtype))
    print(f"Compare compensated output with true output, diff level: {(true_out - compensated_out).abs().mean()}, true output value level: {true_out.abs().mean()}")
    print(f"Compare original output with true output, diff level: {(output - true_out).abs().mean()}, true output value level: {output.abs().mean()}")
    
if __name__ == '__main__':
    
    # test_sparsified_compensation()
    # exit()
    
    a = torch.randn(3, 4, dtype=torch.float16).cuda() * 0.01
    # a = torch.randn(1, 1, 1, 3, 4, dtype=torch.float16).cuda() * 0.01
    # a = torch.tensor([[-0.1, 0.2, -0.3], [0.4, -0.5, 0.6], [-0.7, 0.8, -0.9]]).cuda()
    # a = torch.tensor([[-0.01, 0.48, -0.967], [1.623, -2.222, 2.467], [-2.874, 3.3699, -3.457]]).cuda()
    # a = torch.tensor([[0.001, 0.048, 0.0967], [0.1623, 0.2222, 0.2467], [0.2874, 0.33699, 0.3957]]).cuda()      # e2m1 format
    a = torch.tensor([[0.001, 0.048, 0.0997], [0.1503, 0.2002, 0.2497], [0.2974, 0.30699, 0.4001]]).cuda()      # e2m1 format
    # a = torch.tensor([[0.001, 0.048, 0.0967], [0.1623, 0.2222, 0.2467], [0.2467, 0.2874, 0.2998]]).cuda()       # e1m2 format
    # a = torch.tensor(
    #     [ [ [-0.01,  0.48,   -9.67], 
    #         [1.623,  -2.222, 24.67], ],
    #       [ [-2.874, 3.699,  -34.57], 
    #         [0.85,   -1.343, 18.88], ]
    #     ]
    # ).cuda()        # channel-wise outlier. shape: (2, 2, 3)
    # a = torch.tensor([
    #     [0.1, 0.2, 5.0],
    #     [-0.1, 0.15, -10.0],
    #     [0.0, 0.25, 0.3],
    #     [0.15, -0.2, 0.25],
    # ]).cuda()
    # a = torch.tensor([
    #     [0.1, 0.2, 0.0],
    #     [-0.1, 0.15, 0.0],
    #     [0.0, 0.25, 0.3],
    #     [0.15, -0.2, 0.25],
    # ]).cuda()       # no outlier
    # a = torch.randn(2048, 8096, dtype=torch.bfloat16).cuda() * 0.01     # 模拟1.3B模型的nlp层参数，用于测速

    # data, meta = simu_cast_to_fp4_using_scaling_meta(a)
    # # b = ScalingTensor(data, meta)     # currently not supported, because te not support kFloat4
    # b = data * meta.scale_inv
    
    if False:       # test channel-wise quantization or token-wise quantization
        b = _simu_cast_to_fp4(a, format='e2m1', debug_info=True, channel_wise=True)
        # b = _simu_cast_to_fp4(a, format='e1m2', debug_info=True, token_wise=True, outlier_clip=True, clip_threshold=0.8, nan_existed=True)
        # b = _simu_cast_to_fp4(a.permute(0, 2, 1), format='e1m2', debug_info=True, token_wise=True).permute(0, 2, 1)
        c = b.cast(Dtypes.kfloat8_e4m3)

        print(f"Original tensor: {a}, with max value: {a.abs().max()}")
        print(f"Simulated casted tensor to fp4: {b}, with max value: {b.abs().max()}")
        print(f"Double casted tensor to fp8: {c}")
        print(f"ScaingTensor's data: {c.value - 128}")
    
    elif True:   
        b = _simu_cast_to_fp4(a, format='e2m1', debug_info=True, quantize_method='rtn')
        # b = _simu_cast_to_fp4(a, format='e2m2', debug_info=True, outlier_clip=False, clip_threshold=0.97, nan_existed=True)
        # c = a.cast(Dtypes.kfloat8_e4m3)
        d = b.cast(Dtypes.kfloat8_e4m3)
        
        print(f"Original tensor: {a}, with max value: {a.abs().max()}")
        print(f"Simulated casted tensor to fp8: {b}, with max value: {b.abs().max()}")
        # print(f"MS-AMP casted tensor to fp8: {c}")
        # print(f"ScaingTensor's data: {c.value - 128}")
        print(f"Double casted tensor to fp8: {d}")
        
    elif False:
        # test residual
        large_tensor = torch.randn(1000, 1000, dtype=torch.float16).cuda() * 0.01
        # large_tensor = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=torch.float16).cuda()
        fp4_large_tensor, residual = _simu_cast_to_fp4(large_tensor, format='e2m1', debug_info=True, residual_compensation=True, outlier_clip=True, clip_threshold=0.98)
        # 检查residual张量的形状和稀疏性
        print(f"Original tensor: {large_tensor.shape}, with max value: {large_tensor.abs().max()}")
        print(f"fp4 tensor: {fp4_large_tensor.shape}, with max value: {fp4_large_tensor.abs().max()}")
        print(f"Residual tensor: {residual.shape}, with max value: {residual.abs().max()}")
        print(f"Residual tensor's sparsity: {torch.sum(residual == 0).item() / residual.numel()}")
        
        # test residual + sparse matrix multiplication
        # residual_sparse 
        
    else:
        aa = (2000*a).cast(Dtypes.kfloat8_e4m3)
        print(aa)
        # fp_max = 4.0
        # margin = 0
        # for amax in range(5, 200, 5):
        #     amax = torch.tensor(0.01 * amax).cuda()
        #     sf = ScalingMeta.compute_scaling_factor(amax, torch.ones((), device='cuda'), fp_max, margin)
        #     print(f"amax: {amax:.2f}, computed scaling factor: {sf}")
