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

		void initialize(uint32_t width, uint32_t height);
		void destroy();

		void resize(uint32_t width, uint32_t height);

		DeviceTextureBuffer enableCudaAccess();
		void disableCudaAccess(const DeviceTextureBuffer& dev_tex_buff);

		bool isInitialised() const;

		void copyTo(GLuint dst) const;
		void copyTo(const TextureBuffer& dst) const;

		uint2 getDimensions() const { return m_dimensions; }
		void setDimensions(uint2 size) { m_dimensions = size; }

		GLuint getGLTexture() const { return m_GL_texture; }
		void setTexture(GLuint name) { m_GL_texture = name; }

		cudaGraphicsResource* getGraphicsResource() const { return m_graphics_resource; }
		void setGraphicsResource(cudaGraphicsResource* ptr) { m_graphics_resource = ptr; }

	private:
		
		uint2 m_dimensions{ 0u,0u };
		GLuint m_GL_texture = NULL;//OpenGL texture name
		cudaGraphicsResource* m_graphics_resource = nullptr;//handle to registered OpenGL object
	};
}