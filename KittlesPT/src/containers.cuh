#pragma once
#include "shaders/device_texture_buffer.cuh"
#include <cuda_runtime.h>

namespace KittlesPT
{
	struct GlobalShaderData
	{
		int2 frame_resolution;
		int frame_index = 0;
		float frame_delta = 0.0f;
		DeviceTextureBuffer main_texture;
	};
};