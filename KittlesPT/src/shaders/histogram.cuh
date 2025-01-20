#pragma once
#include "containers.cuh"
#include <device_launch_parameters.h>
#include <vector_types.h>

namespace KittlesPT
{
	__device__ uint colorToBin(float3 hdrColor, float minLogLum, float inverseLogLumRange);
	__device__ float meteringWeight(const GlobalShaderData& shader_data, int2 pixel_coord);
}
//Launch with thread dims 16x16=256
__global__ void histogramComputeKernel(const KittlesPT::GlobalShaderData shader_data);

//Launch with thread dims 256 x 1;
__global__ void histogramAverageLuminanceComputeKernel(const KittlesPT::GlobalShaderData shader_data);