# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# added ruizhe 2024/12/20

"""Tests for FP4 quantization operator."""

import itertools
import unittest

import torch

from tests.helper import decorator
from msamp.operators.fp4_quant import FP4_QUANT


class FP4QuantTestCase(unittest.TestCase):
    '''A class for FP4 quantization test cases.'''
    @decorator.cuda_test
    def test_DGE(self):
        '''Check the DGE item.'''
        from msamp.nn.functional import _differentiable_quantize_derivative
        import time
        
        total_points = 1000
        x_values = torch.linspace(-6.0, 6.0, total_points).to(torch.bfloat16).cuda()
        
        print(f"start benchmark")
        for i in range(15):
            if i == 5:
                time0 = time.time()    # warmup
            differentiable_quantized_y_derivative = FP4_QUANT.apply_DGE_item(x_values, k=3.0, power_clamp_max=3.0)
        print(f"cuda time: {time.time()-time0}")
        for i in range(15):
            if i == 5:
                time0 = time.time()
            differentiable_quantized_y_derivative = _differentiable_quantize_derivative(x_values, k=3.0, power_clamp_max=3.0, level_format='e2m1', nan_existed=False)
        print(f"py time: {time.time()-time0}")
        
        eq_ratio = torch.sum(torch.isclose(differentiable_quantized_y_derivative.float(), differentiable_quantized_y_derivative.float(), atol=1e-6)).item() / differentiable_quantized_y_derivative.numel()
        print(f"eq ratio: {eq_ratio}")
        
        if False:
            import matplotlib.pyplot as plt
            x_values = x_values.float().cpu()
            differentiable_quantized_y_derivative = differentiable_quantized_y_derivative.float().cpu()
            plt.figure(figsize=(8, 6))
            plt.plot(x_values.numpy(), torch.zeros_like(x_values).numpy(), label='Derivative of Hard Quantization', color='blue', linestyle='--')
            plt.plot(x_values.numpy(), differentiable_quantized_y_derivative.numpy(), label='Differentiable Quantization Derivative', color='red')
            plt.plot(x_values.numpy(), torch.ones_like(x_values).numpy(), label='Derivative of Straight-Through Estimator', color='green', linestyle='--', alpha=0.5)
            plt.title('CUDA Differentiable Quantization Derivative (k=3)')
            plt.xlabel('x')
            plt.ylabel('dy/dx')
            plt.grid(True)
            plt.legend()
            plt.savefig("./test_DGE.png")
            plt.close()
    # python -m unittest tests.operators.test_fp4_quant.FP4QuantTestCase.test_DGE
        
    @decorator.cuda_test
    def simple_check(self):
        '''Check the quantization of input tensor.'''
        input_tensor = torch.tensor([[[0.001, 0.048, 0.0997], [0.1503, 0.2002, 0.2497], [0.2974, 0.30699, 0.4001]]], dtype=torch.bfloat16).cuda()
        output_tensor = FP4_QUANT.quantize_simu_fp4_in_bf16(input_tensor, debug_info=True)
        
        print(f"tensor wise quantization")
        print(f"input_tensor: {input_tensor}")
        print(f"output_tensor: {output_tensor}")
        
        input_tensor = torch.tensor(
            [ [ [-0.01,  0.48,   -9.67], 
                [1.623,  -2.222, 24.67], ],
              [ [-2.874, 3.699,  -34.57], 
                [0.85,   -1.343, 18.88], ]
            ], dtype=torch.bfloat16).cuda()        # channel-wise outlier. shape: (2, 2, 3)
        output_tensor = FP4_QUANT.quantize_simu_fp4_in_bf16(input_tensor, channel_wise=True, debug_info=True)
        # output_tensor = FP4_QUANT.quantize_simu_fp4_in_bf16(input_tensor, channel_wise=True, debug_info=True, outlier_clip=True, clip_threshold=0.5)
        
        print(f"channel wise quantization")
        print(f"input_tensor: {input_tensor}")
        print(f"output_tensor: {output_tensor}")
    # python -m unittest tests.operators.test_fp4_quant.FP4QuantTestCase.simple_check
    
    @decorator.cuda_test
    def speed_bench_test(self):
        '''Check the speed of quantization.'''
        from msamp.nn.functional import _simu_cast_to_fp4
        import time
        
        input_size = 8192
        input_tensor = torch.randn(8192, 8192*4, dtype=torch.bfloat16).cuda()
        
        # warmup
        print(f"start benchmark")
        for i in range(15):
            if i == 5:
                time0 = time.time()     # warmup
            #! TODO:
            # output_tensor_cuda = FP4_QUANT.quantize_simu_fp4_in_bf16(input_tensor, format='e1m2', channel_wise=True)   # w
            output_tensor_cuda = FP4_QUANT.quantize_simu_fp4_in_bf16(input_tensor, outlier_clip=True, token_wise=True, residual_compensation=True)  # a
        print(f"cuda time: {time.time()-time0}")
        for i in range(15):
            if i == 5:
                time0 = time.time()     # warmup
            #! TODO:
            # output_tensor_py = _simu_cast_to_fp4(input_tensor, format='e1m2', nan_existed=False, channel_wise=True) # w
            output_tensor_py = _simu_cast_to_fp4(input_tensor, format='e2m1', nan_existed=False, outlier_clip=True, token_wise=True, residual_compensation=True)    # a
        print(f"py time: {time.time()-time0}")
        
        eq_ratio = torch.sum(torch.isclose(output_tensor_cuda.float(), output_tensor_py, atol=1e-3)).item() / output_tensor_cuda.numel()
        print(f"eq ratio: {eq_ratio}")

        # 很奇怪的是allclose会报错，但是打印出来的两个tensor基本上是一样的，可能是因为二分查找和直接硬查找在边界处理方式不一样，所以可能有一些数值被一种方法被归到左端点，而被另一种方法归到右端点
        # tensor-wise下有99%的元素都相等
        # channel-wise或token-wise下，只有90%左右元素相近（还不是相等），可能是由于cuda版本代码直接用bf16去计算了amax和scaling factor
        
    # python -m unittest tests.operators.test_fp4_quant.FP4QuantTestCase.speed_bench_test
        