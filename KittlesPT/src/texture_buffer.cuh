#pragma once

#include "glad/glad.h"
#include <cstdint>
#include <cuda_gl_interop.h>

namespace KittlesPT
{
	struct DeviceTextureBuffer;

	class TextureBuffer
	{
	public:

		void init(uint32_t width, uint32_t height);
		void destroy();

		void resize(uint32_t width, uint32_t height);

		DeviceTextureBuffer enableCudaAccess();
		void disableCudaAccess(const DeviceTextureBuffer& dev_tex_buff);

		bool isInitialised() const;

		void copyTo(GLuint dst) const;
		void copyTo(const TextureBuffer& dst) const;

	public:

		uint32_t m_width = 0, m_height = 0;
		GLuint m_GL_texture = NULL;//OpenGL texture name
		cudaGraphicsResource* m_graphics_resource = nullptr;//handle to registered OpenGL object
	};
}