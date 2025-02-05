#pragma once
#include "containers.cuh"
#include <device_launch_parameters.h>
#include <vector_types.h>

namespace KittlesPT
{
	__device__ float4 textureRead36Texels(const DeviceTextureBuffer& t_texture, float2 t_pixel_coord, bool karis_avg);
	__device__ float4 textureRead36TexelsUV(const DeviceTextureBuffer& t_texture, float2 uv_coord, bool karis_avg);
	__device__ float4 karisAverage(float4 sp0, float4 sp1, float4 sp2, float4 sp3);
}/*KittlesPT*/

__global__ void downSample(const KittlesPT::ShaderData shader_data, KittlesPT::DeviceTextureBuffer t_src, KittlesPT::DeviceTextureBuffer t_dst, bool karis_avg);

__global__ void upSampleCombine(const KittlesPT::ShaderData shader_data, KittlesPT::DeviceTextureBuffer t_src, KittlesPT::DeviceTextureBuffer t_dst);