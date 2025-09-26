#include "texture.hpp"
#include "stb/stb_image.h"

namespace SampleApp
{
	void GLTexture::load(int32_t width, int32_t height, uint8_t* data)
	{
		m_width = width; m_height = height;
		glGenTextures(1, &m_GL_texture_name);
		glBindTexture(GL_TEXTURE_2D, m_GL_texture_name);

		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);

		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, m_width, m_height, 0, GL_RGBA, GL_UNSIGNED_BYTE, data);

		glBindTexture(GL_TEXTURE_2D, 0);
	}

	void GLTexture::init(int32_t width, int32_t height)
	{
		m_width = width; m_height = height;
		glGenTextures(1, &m_GL_texture_name);
		glBindTexture(GL_TEXTURE_2D, m_GL_texture_name);

		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);

		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, m_width, m_height, 0, GL_RGBA, GL_FLOAT, NULL);

		glBindTexture(GL_TEXTURE_2D, 0);
	}

	void GLTexture::resize(int32_t width, int32_t  height)
	{
		if (m_width == width && m_height == height) {
			return;
		}
		m_width = width; m_height = height;

		glBindTexture(GL_TEXTURE_2D, m_GL_texture_name);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, m_width, m_height, 0, GL_RGBA, GL_FLOAT, NULL);
		glBindTexture(GL_TEXTURE_2D, 0);
	}

	//NOTE: this texture must not be resized
	GLTexture loadImage(const char* file_path)
	{
		int width, height, channel_num;
		uint8_t* data = stbi_load(file_path, &width, &height, &channel_num, 4);
		GLTexture texture;
		texture.load(width, height, data);
		return texture;
	}
}