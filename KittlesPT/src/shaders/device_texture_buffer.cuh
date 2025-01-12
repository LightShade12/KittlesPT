#pragma once

#include <glad/glad.h>
#include <cuda_gl_interop.h>

namespace KittlesPT
{
	//TODO: refactor this nonsense

	struct DeviceTextureBuffer
	{
	public:

		__device__ void textureWrite(float4 value, int2 pixel_coord) const;

		__device__ void textureWriteUV(float4 value, float2 uv_coord) const;

		__device__ float4 textureReadNearest(float2 pixel_coord) const;

		__device__ float4 textureReadNearestUV(float2 uv_coord) const;

		__device__ float4 textureReadBilinear(float2 pixel_coord, float filter_alpha) const;

		__device__ float4 textureReadBilinearUV(float2 uv_coord, float filter_alpha) const;

		int2 dimensions;
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

		void copyTo(GLuint dst);

		void copyTo(const TextureBuffer& dst);

	public:

		int m_width = 0, m_height = 0;
		GLuint m_GL_texture = NULL;
		cudaGraphicsResource* m_graphics_resource = nullptr;
	};
}