#pragma once
#include <vector_types.h>
#include <device_launch_parameters.h>

namespace KittlesPT
{
	struct GlobalShaderData;

	struct ShadingJob
	{
		int2 pixel_coord;
		float2 uv_coord;
		bool invalid = false;
	};

	__device__ ShadingJob getShadingJob(const GlobalShaderData& shader_data);
}