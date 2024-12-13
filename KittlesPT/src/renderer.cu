#include "renderer.hpp"

#include "maths/vector_maths.cuh"
#include "shaders/device_texture_buffer.cuh"
#include "containers.cuh"
#include "shaders/kernels.cuh"

#include <unordered_map>
#include <string>
#include <iostream>

namespace KittlesPT
{
	struct RendererData
	{
		GlobalShaderData shader_global_data;
		std::unordered_map< std::string, TextureBuffer>m_frame_textures;
	};

	void Renderer::init()
	{
		m_renderer_data = new RendererData();
		m_renderer_data->m_frame_textures["main_texture"] = TextureBuffer();
	}

	void Renderer::resizeFrame(int width, int height)
	{
		if (m_width == width && m_height == height)
		{
			return;
		}

		m_width = width; m_height = height;
		m_renderer_data->shader_global_data.frame_resolution = make_int2(m_width, m_height);

		if (m_renderer_data->m_frame_textures["main_texture"].isInitialised())
		{
			m_renderer_data->m_frame_textures["main_texture"].resize(m_width, m_height);
		}
		else
		{
			m_renderer_data->m_frame_textures["main_texture"].init(m_width, m_height);
		}
	}
	void Renderer::shutdown()
	{
		for (auto tex : m_renderer_data->m_frame_textures)
		{
			tex.second.destroy();
		}

		delete m_renderer_data;
	}
	void Renderer::executeRendering()
	{
		m_renderer_data->shader_global_data.main_texture = m_renderer_data->m_frame_textures["main_texture"].enableCudaAccess();

		launchRenderPassKernel(m_renderer_data->shader_global_data);

		m_renderer_data->m_frame_textures["main_texture"].disableCudaAccess(m_renderer_data->shader_global_data.main_texture);
	}

	void Renderer::getRenderTargetTexture(GLuint r_texture)
	{
		m_renderer_data->m_frame_textures["main_texture"].copyTo(r_texture);
	}
}