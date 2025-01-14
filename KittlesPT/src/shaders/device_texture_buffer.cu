#include "device_texture_buffer.cuh"

#include "error_check.cuh"
#include "maths/vector_maths.cuh"

#include <cuda.h>

//#define __CUDACC__
//#include <device_functions.h>

#include <cuda_runtime_api.h>//should be used instead of device_functions.h

#include <iostream>

namespace KittlesPT
{
	//====================================================================================
	//DEVICE TEXTURE BUFFER
	//=========================================================================================

	__device__ void DeviceTextureBuffer::textureWrite(float4 value, int2 pixel_coord) const
	{
		surf2Dwrite<float4>(value, m_surface_object, pixel_coord.x * (int)sizeof(float4), pixel_coord.y, cudaBoundaryModeZero);
	}

	__device__ void DeviceTextureBuffer::textureWriteUV(float4 value, float2 uv_coord) const
	{
		int2 pixel_coord = make_int2(uv_coord * dimensions);
		surf2Dwrite<float4>(value, m_surface_object, pixel_coord.x * (int)sizeof(float4), pixel_coord.y, cudaBoundaryModeZero);
	}

	__device__ float4 DeviceTextureBuffer::textureReadNearest(float2 pixel_coord) const
	{
		return surf2Dread<float4>(m_surface_object, int(pixel_coord.x) * (int)sizeof(float4), int(pixel_coord.y), cudaBoundaryModeClamp);
	}

	__device__ float4 DeviceTextureBuffer::textureReadNearestUV(float2 uv_coord) const
	{
		int2 pixel_coord = make_int2(uv_coord * dimensions);
		//pixel_coord = clamp(pixel_coord, make_int2(0, 0), dimensions - 1);
		return surf2Dread<float4>(m_surface_object, pixel_coord.x * (int)sizeof(float4), pixel_coord.y, cudaBoundaryModeClamp);
	}

	__device__ float4 DeviceTextureBuffer::textureReadBilinear(float2 pixel_coord, float filter_alpha) const
	{
		/*
		* Coordinates(verified with uv shading):
		* +1
		* ^
		* ||
		* 0 ==> +1
		*/

		//TODO:consider half pixel for centre sampling

		int2 discrete = make_int2(pixel_coord);

		//tap coord components
		int s0 = discrete.x;
		int s1 = discrete.x + 1;
		int t0 = discrete.y;
		int t1 = discrete.y + 1;

		float ws = pixel_coord.x - discrete.x;
		float wt = pixel_coord.y - discrete.y;

		float4 cp0 = textureReadNearest(make_float2(s0, t0));
		float4 cp1 = textureReadNearest(make_float2(s1, t0));
		float4 cp2 = textureReadNearest(make_float2(s0, t1));
		float4 cp3 = textureReadNearest(make_float2(s1, t1));

		float4 tc0 = lerp(cp0, cp1, ws);
		float4 tc1 = lerp(cp2, cp3, ws);
		float4 fc = lerp(tc0, tc1, wt);

		// point sampling for alpha
		if (!filter_alpha) {
			fc.w = cp0.w;
		}

		return fc;
	}

	__device__ float4 DeviceTextureBuffer::textureReadBilinearUV(float2 uv_coord, float filter_alpha) const
	{
		float2 pixel_coord = uv_coord * dimensions;
		return textureReadBilinear(pixel_coord, filter_alpha);
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
		dev_tex_buf.dimensions = make_int2(m_width, m_height);

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
		GLenum err = glGetError();
		if (err != GL_NO_ERROR) {
			fprintf(stderr, "[DESTROY]: %s\n", glErrorString(err));
		}
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
			fprintf(stderr, "[TEXCOPY_TO_NAME]: %s\n", glErrorString(err));
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
			fprintf(stderr, "[TEXCOPY_TO_OBJECT]: %s\n", glErrorString(err));
		}
	}
}/*KittlesPT*/