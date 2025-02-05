#include "texture_buffer.cuh"
#include "shaders/device_texture_buffer.cuh"
#include "error_check.cuh"
#include <iostream>

namespace KittlesPT
{
	void TextureBuffer::init(uint32_t width, uint32_t height)
	{
		m_width = width; m_height = height;

		glGenTextures(1, &m_GL_texture);
		glBindTexture(GL_TEXTURE_2D, m_GL_texture);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, width, height, 0, GL_RGBA, GL_FLOAT, NULL);

		cudaError_t err = cudaGraphicsGLRegisterImage(&m_graphics_resource, m_GL_texture,
			GL_TEXTURE_2D, cudaGraphicsRegisterFlagsSurfaceLoadStore);
		if (!err == cudaSuccess) {
			printf("Error creating texture\n");
		}

		glBindTexture(GL_TEXTURE_2D, 0);
	}

	void TextureBuffer::destroy()
	{
		cudaGraphicsUnregisterResource(m_graphics_resource);
		m_graphics_resource = nullptr;
		glDeleteTextures(1, &m_GL_texture);
		GLenum err = glGetError();
		if (err != GL_NO_ERROR) {
			fprintf(stderr, "[TEX DESTROY]: %s\n", glErrorString(err));
		}
		m_GL_texture = NULL;
	}

	void TextureBuffer::resize(uint32_t width, uint32_t height)
	{
		m_width = width; m_height = height;

		cudaGraphicsUnregisterResource(m_graphics_resource);
		glBindTexture(GL_TEXTURE_2D, m_GL_texture);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, m_width, m_height, 0, GL_RGBA, GL_FLOAT, NULL);
		glBindTexture(GL_TEXTURE_2D, 0);
		cudaGraphicsGLRegisterImage(&m_graphics_resource, m_GL_texture, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsSurfaceLoadStore);
	}

	DeviceTextureBuffer TextureBuffer::enableCudaAccess()
	{
		cudaGraphicsMapResources(1, &m_graphics_resource, 0);
		cudaResourceDesc resource_descriptor;
		resource_descriptor.resType = cudaResourceTypeArray;
		cudaGraphicsSubResourceGetMappedArray(&resource_descriptor.res.array.array, m_graphics_resource,
			0, 0);

		DeviceTextureBuffer dev_tex_buf;
		dev_tex_buf.dimensions = make_int2(m_width, m_height);
		cudaCreateSurfaceObject(&dev_tex_buf.m_surface_object, &resource_descriptor);

		return dev_tex_buf;
	}

	void TextureBuffer::disableCudaAccess(const DeviceTextureBuffer& dev_tex_buff)
	{
		cudaDestroySurfaceObject(dev_tex_buff.m_surface_object);
		cudaGraphicsUnmapResources(1, &m_graphics_resource, 0);
	}

	bool TextureBuffer::isInitialised() const
	{
		return (m_GL_texture != NULL && m_graphics_resource != nullptr);
	}

	void TextureBuffer::copyTo(GLuint dst) const
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
	void TextureBuffer::copyTo(const TextureBuffer& dst) const
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
}