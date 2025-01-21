#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cmath>
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>

#include "../../common/include/common.h"


// FP4_E2M1_no_NaN硬查找式量化
__global__ void quantize_bf16_kernel(const __nv_bfloat16* x, __nv_bfloat16* output, int x_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < x_size) {
        __nv_bfloat16 value = x[idx];
        __nv_bfloat16 closest;

        if (__hlt(value, __float2bfloat16(-5.0f))) {
            closest = __float2bfloat16(-6.0f);
        } else if (__hlt(value, __float2bfloat16(-3.5f))) {
            closest = __float2bfloat16(-4.0f);
        } else if (__hlt(value, __float2bfloat16(-2.5f))) {
            closest = __float2bfloat16(-3.0f);
        } else if (__hlt(value, __float2bfloat16(-1.75f))) {
            closest = __float2bfloat16(-2.0f);
        } else if (__hlt(value, __float2bfloat16(-1.25f))) {
            closest = __float2bfloat16(-1.5f);
        } else if (__hlt(value, __float2bfloat16(-0.75f))) {
            closest = __float2bfloat16(-1.0f);
        } else if (__hlt(value, __float2bfloat16(-0.25f))) {
            closest = __float2bfloat16(-0.5f);
        } else if (__hlt(value, __float2bfloat16(0.25f))) {
            closest = __float2bfloat16(0.0f);
        } else if (__hlt(value, __float2bfloat16(0.75f))) {
            closest = __float2bfloat16(0.5f);
        } else if (__hlt(value, __float2bfloat16(1.25f))) {
            closest = __float2bfloat16(1.0f);
        } else if (__hlt(value, __float2bfloat16(1.75f))) {
            closest = __float2bfloat16(1.5f);
        } else if (__hlt(value, __float2bfloat16(2.5f))) {
            closest = __float2bfloat16(2.0f);
        } else if (__hlt(value, __float2bfloat16(3.5f))) {
            closest = __float2bfloat16(3.0f);
        } else if (__hlt(value, __float2bfloat16(5.0f))) {
            closest = __float2bfloat16(4.0f);
        } else {
            closest = __float2bfloat16(6.0f);
        }

        output[idx] = closest;
    }
}

// FP4量化主机接口函数
void quantize_bf16(at::Tensor input, at::Tensor output, int size) {

    const __nv_bfloat16* input_data = reinterpret_cast<const __nv_bfloat16*>(input.data_ptr<at::BFloat16>());  
    __nv_bfloat16* output_data = reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()); 
    
    // 配置线程和块
    const int threadsPerBlock = HIP_GET_NUM_THREADS(size);              // 512
    // const int blocks = HIP_GET_BLOCKS(size, threadsPerBlock);        // 该函数规定了最大grid num为HIP_MAX_GRID_NUM = 65535，在处理很大的size时会出现问题
    const int blocks = (size + threadsPerBlock - 1) / threadsPerBlock;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // 调用核函数
    quantize_bf16_kernel<<<blocks, threadsPerBlock, 0, stream>>>(input_data, output_data, size);

    // 同步以确保执行完成
    // cudaDeviceSynchronize();
}

// 可微幂近似函数的导数
__device__ float power_derivative(float x, float delta, float k, float power_clamp_max) {
    float abs_term = fabsf(2.0f * x / delta - 1.0f);
    return fminf(powf(abs_term, 1.0f / k - 1.0f) / k, power_clamp_max);
}

__device__ float power_derivative_delta2(float x){
    return fminf(powf(fabsf(x - 1.0f), - 0.6666667f) / 3.0f, 3.0f);
}

__device__ float power_derivative_delta1(float x){
    return fminf(powf(fabsf(2.0f * x - 1.0f), - 0.6666667f) / 3.0f, 3.0f);
}

__device__ float power_derivative_delta05(float x){
    return fminf(powf(fabsf(4.0f * x - 1.0f), - 0.6666667f) / 3.0f, 3.0f);
}

// 计算可微幂近似函数的导数的CUDA核函数
// 量化区间固定为[-6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]，即E2M1_no_NaN
__global__ void differentiable_quantize_derivative(
    const __nv_bfloat16* input, __nv_bfloat16* output
    float k, float power_clamp_max, int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float x = __bfloat162float(input[idx]);
    float dy = 0.0f;

    // 硬编码区间索引匹配
    if (x >= -6.0f && x < -4.0f) {
        dy = power_derivative(x + 6.0f, 2.0f, k, power_clamp_max);
    } else if (x >= -4.0f && x < -3.0f) {
        dy = power_derivative(x + 4.0f, 1.0f, k, power_clamp_max);
    } else if (x >= -3.0f && x < -2.0f) {
        dy = power_derivative(x + 3.0f, 1.0f, k, power_clamp_max);
    } else if (x >= -2.0f && x < -1.5f) {
        dy = power_derivative(x + 2.0f, 0.5f, k, power_clamp_max);
    } else if (x >= -1.5f && x < -1.0f) {
        dy = power_derivative(x + 1.5f, 0.5f, k, power_clamp_max);
    } else if (x >= -1.0f && x < -0.5f) {
        dy = power_derivative(x + 1.0f, 0.5f, k, power_clamp_max);
    } else if (x >= -0.5f && x < 0.0f) {
        dy = power_derivative(x + 0.5f, 0.5f, k, power_clamp_max);
    } else if (x >= 0.0f && x < 0.5f) {
        dy = power_derivative(x, 0.5f, k, power_clamp_max);
    } else if (x >= 0.5f && x < 1.0f) {
        dy = power_derivative(x - 0.5f, 0.5f, k, power_clamp_max);
    } else if (x >= 1.0f && x < 1.5f) {
        dy = power_derivative(x - 1.0f, 0.5f, k, power_clamp_max);
    } else if (x >= 1.5f && x < 2.0f) {
        dy = power_derivative(x - 1.5f, 0.5f, k, power_clamp_max);
    } else if (x >= 2.0f && x < 3.0f) {
        dy = power_derivative(x - 2.0f, 1.0f, k, power_clamp_max);
    } else if (x >= 3.0f && x < 4.0f) {
        dy = power_derivative(x - 3.0f, 1.0f, k, power_clamp_max);
    } else if (x >= 4.0f && x <= 6.0f) {
        dy = power_derivative(x - 4.0f, 2.0f, k, power_clamp_max);
    }

    output[idx] = __float2bfloat16(dy);
}

// 可微幂近似函数的导数主机接口函数
void launch_differentiable_quantize_derivative(
    at::Tensor input, at::Tensor output,
    float k, float power_clamp_max, int size
) {
    const __nv_bfloat16* input_data = reinterpret_cast<const __nv_bfloat16*>(input.data_ptr<at::BFloat16>());  
    __nv_bfloat16* output_data = reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()); 
    
    // 配置线程和块
    const int threadsPerBlock = HIP_GET_NUM_THREADS(size);              // 512
    // const int blocks = HIP_GET_BLOCKS(size, threadsPerBlock);        // 该函数规定了最大grid num为HIP_MAX_GRID_NUM = 65535，在处理很大的size时会出现问题
    const int blocks = (size + threadsPerBlock - 1) / threadsPerBlock;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    differentiable_quantize_derivative<<<blocks, threadsPerBlock, 0, stream>>>(
        input_data, output_data, k, power_clamp_max, size
    );
    // cudaDeviceSynchronize();
}

// Pybind11接口
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("quantize_bf16", &quantize_bf16, "Simulated Quantize FP4 Function in BF16 Format");
    m.def("launch_differentiable_quantize_derivative", &launch_differentiable_quantize_derivative, "Differentiable Quantize Derivative Function");
}