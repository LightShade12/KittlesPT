#pragma once
#include "shaders/device_texture_buffer.cuh"
#include "shaders/sphere.cuh"
#include "shaders/camera.cuh"
#include <cuda_runtime.h>

namespace KittlesPT
{
	template <typename T> struct Buffer
	{
		Buffer() = default;
		Buffer(T* data_, size_t num_)
			:data(data_), num(num_) {};

		T* data = nullptr;
		size_t num = 0;
	};

	struct GlobalShaderData
	{
		int2 frame_resolution;
		int frame_index = 0;
		float frame_delta = 0.0f;
		Buffer<Sphere> scene_buffer;
		Camera scene_camera;
		DeviceTextureBuffer main_texture;
		DeviceTextureBuffer accumulation_texture;
	};
};