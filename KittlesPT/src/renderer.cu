#include "renderer.hpp"

#include "maths/vector_maths.cuh"
#include "containers.cuh"
#include "shaders/device_texture_buffer.cuh"
#include "shaders/kernels.cuh"
#include "shaders/material.cuh"

#include "glm/glm.hpp"
#include "glm/gtc/matrix_transform.hpp"

#include <thrust/universal_vector.h>

#include <unordered_map>
#include <string>
#include <iostream>

namespace KittlesPT
{
	struct RendererData
	{
		thrust::universal_vector<Sphere> scene_spheres;
		thrust::universal_vector<Material> scene_materials;
		GlobalShaderData shader_global_data;
		std::unordered_map< std::string, TextureBuffer>m_frame_textures;
	};

	void Renderer::init()
	{
		int cuda_driver_version, cuda_runtime_version;
		cudaDriverGetVersion(&cuda_driver_version); cudaRuntimeGetVersion(&cuda_runtime_version);
		printf("CUDA driver version: %d.%d\nCUDA toolkit runtime version: %d.%d\n",
			cuda_driver_version / 1000, cuda_driver_version % 100, cuda_runtime_version / 1000, cuda_runtime_version % 100);
		m_renderer_data = new RendererData();
		m_renderer_data->m_frame_textures["main_texture"] = TextureBuffer();
		m_renderer_data->m_frame_textures["accumulation_texture"] = TextureBuffer();

		m_renderer_data->scene_spheres.push_back(Sphere(0.5, make_float3(-1.5, 0, -3), 0));
		m_renderer_data->scene_spheres.push_back(Sphere(0.5, make_float3(0, 0, -3), 2));
		m_renderer_data->scene_spheres.push_back(Sphere(0.5, make_float3(1.5, 0, -3), 3));
		m_renderer_data->scene_spheres.push_back(Sphere(100, make_float3(0, -100.5, -3), 1));

		m_renderer_data->scene_materials.push_back(Material(
			make_float3(0.95, 0.1, 0.1),
			0.0,
			0.1,
			0.0f));

		m_renderer_data->scene_materials.push_back(Material(
			make_float3(0.8, 0.8, 0.8),
			0.0,
			0.8,
			0.0f));

		m_renderer_data->scene_materials.push_back(Material(
			make_float3(0.8, 0.8, 0.8),
			1.0,
			0.2,
			0.0f));

		m_renderer_data->scene_materials.push_back(Material(
			make_float3(0.9, 0.9, 0.9),
			0.0,
			0.1,
			1.0f));

		//submit

		m_renderer_data->shader_global_data.scene_buffer =
			Buffer<Sphere>(thrust::raw_pointer_cast(
				m_renderer_data->scene_spheres.data()),
				m_renderer_data->scene_spheres.size());

		m_renderer_data->shader_global_data.materials_buffer =
			Buffer<Material>(thrust::raw_pointer_cast(
				m_renderer_data->scene_materials.data()),
				m_renderer_data->scene_materials.size());

		m_renderer_data->shader_global_data.scene_camera = Camera(make_float3(0), make_float3(0, 0, -1));
	}
	void Renderer::shutdown()
	{
		for (auto tex : m_renderer_data->m_frame_textures)
		{
			tex.second.destroy();
		}

		m_renderer_data->scene_spheres.clear();//TODO: put this in destroy/destructor
		m_renderer_data->scene_materials.clear();//TODO: put this in destroy/destructor
		delete m_renderer_data;
	}

	void Renderer::resizeFrame(int width, int height)
	{
		if (m_width == width && m_height == height)
		{
			return;
		}

		m_width = width; m_height = height;
		m_renderer_data->shader_global_data.frame_resolution = make_int2(m_width, m_height);
		glm::mat4 view = glm::mat4
		(1, 0, 0, 0,
			0, 1, 0, 0,
			0, 0, -1, 0,
			0, 0, 0, 1);
		setView(glm::perspectiveFovLH(glm::radians(90.0f),
			float(m_width), float(m_height), 1.f, 100.f),
			glm::inverse(view));

		if (m_renderer_data->m_frame_textures["main_texture"].isInitialised())
		{
			m_renderer_data->m_frame_textures["main_texture"].resize(m_width, m_height);
			m_renderer_data->m_frame_textures["accumulation_texture"].resize(m_width, m_height);
		}
		else
		{
			m_renderer_data->m_frame_textures["main_texture"].init(m_width, m_height);
			m_renderer_data->m_frame_textures["accumulation_texture"].init(m_width, m_height);
		}
	}
	void Renderer::executeRendering()
	{
		m_renderer_data->shader_global_data.main_texture = m_renderer_data->m_frame_textures["main_texture"].enableCudaAccess();
		m_renderer_data->shader_global_data.accumulation_texture = m_renderer_data->m_frame_textures["accumulation_texture"].enableCudaAccess();

		launchRenderPassKernel(m_renderer_data->shader_global_data);

		m_renderer_data->m_frame_textures["main_texture"].disableCudaAccess(m_renderer_data->shader_global_data.main_texture);
		m_renderer_data->m_frame_textures["accumulation_texture"].disableCudaAccess(m_renderer_data->shader_global_data.accumulation_texture);

		m_renderer_data->shader_global_data.frame_index++;//TODO:expose to host as readonly?
	}

	void Renderer::getRenderTargetTexture(GLuint r_texture)
	{
		m_renderer_data->m_frame_textures["main_texture"].copyTo(r_texture);
	}

	void Renderer::setView(glm::mat4 projection_mat, glm::mat4 view_mat)
	{
		//TODO: skip inversion
		Mat4 proj = Mat4(projection_mat);
		Mat4 view = Mat4(view_mat);
		m_renderer_data->shader_global_data.scene_camera.setView(proj.inverse(), view.inverse());
		//reset accum
		glClearTexImage(m_renderer_data->m_frame_textures["accumulation_texture"].m_GL_texture, 0, GL_RGBA, GL_FLOAT, NULL);
		m_renderer_data->shader_global_data.frame_index = 0;
	}
}