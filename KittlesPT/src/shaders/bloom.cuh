#pragma once
#include "containers.cuh"
#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#include <vector_types.h>

namespace KittlesPT
{
	__device__ float4 texRead36Texel(DeviceTextureBuffer t_tex, int2 t_res, int2 t_pixel_coord);
}

__global__ void downSample(const KittlesPT::GlobalShaderData t_shader_data, KittlesPT::DeviceTextureBuffer t_src, KittlesPT::DeviceTextureBuffer t_dst);

__global__ void upSampleCombine(const KittlesPT::GlobalShaderData t_shader_data, KittlesPT::DeviceTextureBuffer t_src, KittlesPT::DeviceTextureBuffer t_dst);