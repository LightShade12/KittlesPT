#pragma once
#include "autoexposure_program.hpp"
#include "containers.cuh"
#include "mipchain.cuh"
#include "texture_buffer.cuh"
#include "as_builder.cuh"

#include <thrust/universal_vector.h>
#include <thrust/device_vector.h>

#include <unordered_map>

namespace KittlesPT
{
	//TODO: add API for direct content management

	struct RendererResource
	{
		RendererResource() :
			blas_builder(&scene_triangles, &triangle_index_buffer, &bvhnodes_buffer),
			tlas_builder(&blas_buffer, &tlasnodes_buffer)
		{};

		void updateTLAS() {
			shader_data.top_level_acceleration_structure = tlas_builder.build();
		}

		void updateResource()
		{
			shader_data.tlas_nodes_buffer = Buffer<TLASNode>(thrust::raw_pointer_cast(tlasnodes_buffer.data()), tlasnodes_buffer.size());
			shader_data.blas_buffer = Buffer<BLAS>(thrust::raw_pointer_cast(blas_buffer.data()), blas_buffer.size());
			shader_data.bvh_nodes_buffer = Buffer<BVHNode>(thrust::raw_pointer_cast(bvhnodes_buffer.data()), bvhnodes_buffer.size());
			shader_data.triangle_index_buffer = Buffer<int32_t>(thrust::raw_pointer_cast(triangle_index_buffer.data()), triangle_index_buffer.size());
			shader_data.meshes_buffer = Buffer<TriangleMesh>(thrust::raw_pointer_cast(scene_meshes.data()), scene_meshes.size());
			shader_data.lights_buffer = Buffer<Light>(thrust::raw_pointer_cast(scene_lights.data()), scene_lights.size());
			shader_data.triangles_buffer = Buffer<Triangle>(thrust::raw_pointer_cast(scene_triangles.data()), scene_triangles.size());
			shader_data.materials_buffer = Buffer<Material>(thrust::raw_pointer_cast(scene_materials.data()), scene_materials.size());
			shader_data.texture_buffer = Buffer<Texture>(thrust::raw_pointer_cast(scene_textures.data()), scene_textures.size());
			shader_data.pixel_buffer = Buffer<uint8_t>(thrust::raw_pointer_cast(pixel_buffer.data()), pixel_buffer.size());
		}

		void destroy()
		{
			printf("[RENDERER RESOURCE]: Destroying\n");
			cudaFree(shader_data.scene_average_luminance);
			bloom_mipchain.destroy();

			for (std::pair<const std::string, TextureBuffer>& tex : m_frame_textures)
			{
				tex.second.destroy();
			}

			tlasnodes_buffer.clear();
			blas_buffer.clear();
			bvhnodes_buffer.clear();
			triangle_index_buffer.clear();
			scene_meshes.clear();
			scene_lights.clear();
			scene_triangles.clear();
			scene_materials.clear();
			scene_textures.clear();
			pixel_buffer.clear();

			histogram_buffer.clear();
			m_frame_textures.clear();
		}

		~RendererResource()
		{
			destroy();
		}

		//TODO: store ShaderData in constant_memory
		ShaderData shader_data;
		MipChain bloom_mipchain;
		BLASBuilder blas_builder;
		TLASBuilder tlas_builder;
		AutoExposureProgram auto_exposure_program;

		thrust::universal_vector<TLASNode> tlasnodes_buffer;
		thrust::universal_vector<BLAS> blas_buffer;
		thrust::universal_vector<BVHNode> bvhnodes_buffer;
		thrust::universal_vector<int32_t> triangle_index_buffer;
		thrust::universal_vector<TriangleMesh> scene_meshes;
		thrust::universal_vector<Light> scene_lights;
		thrust::universal_vector<Triangle> scene_triangles;
		thrust::universal_vector<Material> scene_materials;
		thrust::universal_vector<Texture> scene_textures;
		thrust::device_vector<uint8_t> pixel_buffer;

		thrust::device_vector<float> histogram_buffer;
		std::unordered_map< std::string, TextureBuffer>m_frame_textures;
	};
}