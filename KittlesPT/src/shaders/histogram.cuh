#pragma once
#include "containers.cuh"
#include <device_launch_parameters.h>
#include <vector_types.h>

namespace KittlesPT
{
	__device__ uint luminanceToBin(float luminance, float min_log_lum, float inverse_log_lum_range);
	__device__ float centerMeteringWeight(const int2 frame_resolution , int2 pixel_coord, float radius_factor);
}
//Launch with thread dims 16x16=256
__global__ void histogramComputeKernel(const KittlesPT::ShaderData shader_data);

//Launch with thread dims 256 x 1;
__global__ void histogramAverageLuminanceComputeKernel(const KittlesPT::ShaderData shader_data);