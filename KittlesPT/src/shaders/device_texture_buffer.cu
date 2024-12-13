#include "device_texture_buffer.cuh"

#include "../error_check.cuh"
#include "../maths/vector_maths.cuh"

#include <cuda.h>
#define __CUDACC__
#include <cuda_runtime_api.h>//should be used instead of device_fnctions.h
#include <device_functions.h>

#include <iostream>

namespace KittlesPT
{
	//====================================================================================
	//DEVICE TEXTURE BUFFER
	//=========================================================================================

	__device__ void DeviceTextureBuffer::textureWrite(float4 value, int2 pixel_coord) const
	{
		surf2Dwrite<float4>(value, m_surface_object, pixel_coord.x * (int)sizeof(float4), pixel_coord.y);
	}

	//TODO:fix UV coords
	__device__ void DeviceTextureBuffer::textureWrite(float4 value, float2 uv_coord) const
	{
		float2 a = make_float2(width, height) * uv_coord;
		int2 pixel_coord = make_int2(a);
		pixel_coord = clamp(pixel_coord, make_int2(0, 0), make_int2(width - 1, height - 1));
		surf2Dwrite<float4>(value, m_surface_object, pixel_coord.x * (int)sizeof(float4), pixel_coord.y);
	}

	__device__ float4 DeviceTextureBuffer::textureReadNearest(int2 pixel_coord) const
	{
		return surf2Dread<float4>(m_surface_object, pixel_coord.x * (int)sizeof(float4), pixel_coord.y);
	}

	__device__ float4 DeviceTextureBuffer::textureReadNearest(float2 uv_coord) const
	{
		int2 pixel_coord = make_int2(uv_coord.x * width, uv_coord.y * height);
		pixel_coord = clamp(pixel_coord, make_int2(0, 0), make_int2(width - 1, height - 1));
		return surf2Dread<float4>(m_surface_object, pixel_coord.x * (int)sizeof(float4), pixel_coord.y);
	}

	//==================================================================================================
	//TEXTURE BUFFER
	//====================================================================================================

	void TextureBuffer::init(int width, int height)
	{
		m_width = width; m_height = height;
		//GL texture configure
		glGenTextures(1, &m_GL_texture);
		glBindTexture(GL_TEXTURE_2D, m_GL_texture);

		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);

		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, width, height, 0, GL_RGBA, GL_FLOAT, NULL);

		cudaError_t err = cudaGraphicsGLRegisterImage(&m_graphics_resource, m_GL_texture, GL_TEXTURE_2D,
			cudaGraphicsRegisterFlagsSurfaceLoadStore);

		if (!err == cudaSuccess) {
			printf("Error creating texture\n");
		}

		glBindTexture(GL_TEXTURE_2D, 0);
	}
	void TextureBuffer::resize(int width, int height)
	{
		m_width = width; m_height = height;

		cudaGraphicsUnregisterResource(m_graphics_resource);
		glBindTexture(GL_TEXTURE_2D, m_GL_texture);
		{
			glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, m_width, m_height, 0, GL_RGBA, GL_FLOAT, NULL);
		}
		glBindTexture(GL_TEXTURE_2D, 0);
		cudaGraphicsGLRegisterImage(&m_graphics_resource, m_GL_texture,
			GL_TEXTURE_2D, cudaGraphicsRegisterFlagsSurfaceLoadStore);
	}

	DeviceTextureBuffer TextureBuffer::enableCudaAccess()
	{
		cudaGraphicsMapResources(1, &m_graphics_resource, 0);
		cudaArray_t sub_resource_array;
		cudaGraphicsSubResourceGetMappedArray(&sub_resource_array, m_graphics_resource,
			0, 0);
		cudaResourceDesc resource_descriptor;
		{
			resource_descriptor.resType = cudaResourceTypeArray;
			resource_descriptor.res.array.array = sub_resource_array;
		}
		cudaSurfaceObject_t surface;
		cudaCreateSurfaceObject(&surface, &resource_descriptor);

		DeviceTextureBuffer dev_tex_buf;
		dev_tex_buf.m_surface_object = surface;
		dev_tex_buf.height = m_height;
		dev_tex_buf.width = m_width;

		return dev_tex_buf;
	}
	void TextureBuffer::disableCudaAccess(DeviceTextureBuffer dev_tex_buff)
	{
		cudaDestroySurfaceObject(dev_tex_buff.m_surface_object);
		cudaGraphicsUnmapResources(1, &m_graphics_resource, 0);
	}
	bool TextureBuffer::isInitialised()
	{
		return (m_GL_texture != NULL && m_graphics_resource != nullptr);
	}
	void TextureBuffer::destroy()
	{
		cudaGraphicsUnregisterResource(m_graphics_resource);
		m_graphics_resource = nullptr;
		glDeleteTextures(1, &m_GL_texture);
		m_GL_texture = NULL;
	}
	void TextureBuffer::copyTo(GLuint dst)
	{
		glCopyImageSubData(
			m_GL_texture, GL_TEXTURE_2D, 0, 0, 0, 0,
			dst, GL_TEXTURE_2D, 0, 0, 0, 0,
			m_width, m_height, 1);
		glFinish();
		GLenum err = glGetError();
		if (err != GL_NO_ERROR) {
			fprintf(stderr, "[TEXCOPY]: %s\n", glErrorString(err));
		}
	}
	void TextureBuffer::copyTo(const TextureBuffer& dst)
	{
		glCopyImageSubData(
			m_GL_texture, GL_TEXTURE_2D, 0, 0, 0, 0,
			dst.m_GL_texture, GL_TEXTURE_2D, 0, 0, 0, 0,
			m_width, m_height, 1);
		glFinish();
		GLenum err = glGetError();
		if (err != GL_NO_ERROR) {
			fprintf(stderr, "[TEXCOPY]: %s\n", glErrorString(err));
		}
	}
}