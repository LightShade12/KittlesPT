#pragma once
#include "shaders/device_texture_buffer.cuh"
#include <cuda_runtime.h>

namespace KittlesPT
{
	struct GlobalShaderData
	{
		int2 frame_resolution;
		DeviceTextureBuffer main_texture;
	};
};