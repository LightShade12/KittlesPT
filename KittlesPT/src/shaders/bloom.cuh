#pragma once
#include "containers.cuh"
#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#include <vector_types.h>

namespace KittlesPT
{
	__device__ float4 texRead36Texel(DeviceTextureBuffer t_tex, float2 t_pixel_coord, bool karis_avg);
	__device__ float4 texRead36TexelUV(DeviceTextureBuffer t_tex, float2 uv_coord, bool karis_avg);
	__device__ float4 karisAverage(float4 sp0, float4 sp1, float4 sp2, float4 sp3);
}

__global__ void downSample(const KittlesPT::GlobalShaderData t_shader_data, KittlesPT::DeviceTextureBuffer t_src, KittlesPT::DeviceTextureBuffer t_dst, bool karis_avg);

__global__ void upSampleCombine(const KittlesPT::GlobalShaderData t_shader_data, KittlesPT::DeviceTextureBuffer t_src, KittlesPT::DeviceTextureBuffer t_dst);