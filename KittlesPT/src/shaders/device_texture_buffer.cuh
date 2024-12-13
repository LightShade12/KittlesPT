#pragma once

#include <glad/glad.h>
#include <cuda_gl_interop.h>

namespace KittlesPT
{
	struct DeviceTextureBuffer
	{
	public:

		__device__ void textureWrite(float4 value, int2 pixel_coord) const;

		__device__ void textureWrite(float4 value, float2 uv_coord) const;

		__device__ float4 textureReadNearest(int2 pixel_coord) const;

		__device__ float4 textureReadNearest(float2 uv_coord) const;

		int width = 0, height = 0;
		cudaSurfaceObject_t m_surface_object;
	};

	class TextureBuffer
	{
	public:

		void init(int width, int height);

		void resize(int width, int height);

		DeviceTextureBuffer enableCudaAccess();

		void disableCudaAccess(DeviceTextureBuffer dev_tex_buff);

		bool isInitialised();

		void destroy();

		int m_width = 0, m_height = 0;
		GLuint m_GL_texture = NULL;
		cudaGraphicsResource* m_graphics_resource = nullptr;

		void copyTo(GLuint dst);

		void copyTo(const TextureBuffer& dst);
	};
}