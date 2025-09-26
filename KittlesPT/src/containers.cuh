#pragma once
#include "shaders/device_texture_buffer.cuh"
#include "shaders/camera.cuh"
#include "shaders/tlas.cuh"
#include "pod_types.hpp"

namespace KittlesPT
{
	class BVH2Node;
	class TLASNode;
	class BLAS2;
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
		int2 output_resolution;
		int2 render_resolution;
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
		Buffer<BVH2Node> bvh_nodes_buffer;
		Buffer<BLAS2> blas_buffer;
		Buffer<TLASNode> tlas_nodes_buffer;
		TLAS top_level_acceleration_structure;
		Camera scene_camera;

		//custom shader data
		ProceduralEnvironmentSettings procedural_environment_data;
		RendererSettings renderer_settings;

		//frame textures
		DeviceTextureBuffer output_texture;
		DeviceTextureBuffer render_texture;
		DeviceTextureBuffer accumulation_texture;
		DeviceTextureBuffer gbuffer_texture;
		DeviceTextureBuffer prev_gbuffer_texture;
		DeviceTextureBuffer vbuffer_texture;
		DeviceTextureBuffer debug_texture;
		DeviceTextureBuffer backbuffer_texture;
		DeviceTextureBuffer bloom_texture;
	};
};