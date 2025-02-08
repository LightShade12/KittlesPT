#pragma once
#include "shaders/device_texture_buffer.cuh"
#include "shaders/camera.cuh"
#include "shaders/tlas.cuh"
#include "pod_types.hpp"

namespace KittlesPT
{
	class BVHNode;
	class TLASNode;
	class BLAS;
	struct Material;
	class Triangle;
	class TriangleMesh;
	class Light;
	class Texture;

	template <typename T> struct Buffer
	{
		Buffer() = default;
		Buffer(T* data_, size_t num_)
			:data(data_), num(num_) {};

		T* data = nullptr;
		size_t num = 0;
	};

	struct ShaderData
	{
		/* ShaderToy
		uniform vec3 iResolution; viewport res in px
		uniform float iTime; shaderplaybacktime in secs
		uniform float iTimeDelta; render time in secs
		uniform float iFrameRate; fps
		*/

		//standard shader uniforms
		int2 frame_resolution;//output resolution
		int32_t frame_index = 0;
		float frame_delta_ms = 0.0f;//in ms

		//post process uniforms
		float* scene_average_luminance = nullptr;//nits

		//geometry shader uniforms
		Buffer<Triangle> triangles_buffer;
		Buffer<TriangleMesh> meshes_buffer;
		Buffer<float> histogram_buffer;
		Buffer<Material> materials_buffer;
		Buffer<Light> lights_buffer;
		Buffer<Texture> texture_buffer;
		Buffer<uint8_t> pixel_buffer;
		Buffer<int32_t> triangle_index_buffer;
		Buffer<BVHNode> bvh_nodes_buffer;
		Buffer<BLAS> blas_buffer;
		Buffer<TLASNode> tlas_nodes_buffer;
		TLAS top_level_acceleration_structure;
		Camera scene_camera;

		//custom shader data
		ProceduralEnvironmentSettings procedural_environment_data;
		RendererSettings renderer_settings;

		//frame textures
		DeviceTextureBuffer main_texture;
		DeviceTextureBuffer accumulation_texture;
		DeviceTextureBuffer gbuffer_texture;
		DeviceTextureBuffer debug_texture;
		DeviceTextureBuffer bloom_texture;
	};
};