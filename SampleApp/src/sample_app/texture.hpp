#pragma once
#include "glad/glad.h"
#include <cstdint>

namespace SampleApp
{
	struct GLTexture
	{
		void init(int width, int height);

		void resize(int width, int  height);

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
}