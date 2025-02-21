#pragma once
#include "glad/glad.h"
#include <cstdint>

namespace SampleApp
{
	struct GLTexture
	{
		GLTexture() = default;

		void load(int32_t width, int32_t height, uint8_t* data);

		void init(int32_t width, int32_t height);

		void resize(int32_t width, int32_t height);

		bool isValid() const {
			return m_GL_texture_name != NULL;
		}

		void destroy() {
			glDeleteTextures(1, &m_GL_texture_name);
		}

		GLuint getGLTexture() const {
			return m_GL_texture_name;
		}

	private:
		int32_t m_width = 0, m_height = 0;
		GLuint m_GL_texture_name = NULL;
	};

	GLTexture loadImage(const char* file_path);
}