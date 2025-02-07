#pragma once
#include "containers.cuh"
#include <device_launch_parameters.h>
#include <vector_types.h>

__global__ void downSample(const KittlesPT::ShaderData shader_data, KittlesPT::DeviceTextureBuffer t_src,
	KittlesPT::DeviceTextureBuffer t_dst, bool karis_avg);

__global__ void upSampleCombine(const KittlesPT::ShaderData shader_data, KittlesPT::DeviceTextureBuffer t_src,
	KittlesPT::DeviceTextureBuffer t_dst);