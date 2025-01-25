#pragma once
#include "shaders/device_texture_buffer.cuh"
#include "shaders/sphere.cuh"
#include "shaders/camera.cuh"
#include "shaders/material.cuh"
#include "shaders/light.cuh"
#include "shaders/texture.cuh"
#include "pod_types.hpp"

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

		float* scene_average_luminance = nullptr;
		Buffer<Sphere> geometry_buffer;
		Buffer<float> histogram_buffer;
		Buffer<Material> materials_buffer;
		Buffer<Light> lights_buffer;
		Buffer<Texture> texture_buffer;
		Buffer<unsigned char> pixel_buffer;

		Camera scene_camera;

		ProceduralEnvironmentData procedural_environment_data;
		RendererSettings renderer_settings;

		DeviceTextureBuffer main_texture;
		DeviceTextureBuffer accumulation_texture;
		DeviceTextureBuffer gbuffer_texture;
		DeviceTextureBuffer bloom_texture;
	};
};