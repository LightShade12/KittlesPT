#include "texture.hpp"

void GLTexture::init(int width, int height)
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
void GLTexture::resize(int width, int  height)
{
	if (m_width == width && m_height == height)
	{
		return;
	}
	m_width = width; m_height = height;

	glBindTexture(GL_TEXTURE_2D, m_GL_texture_name);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, m_width, m_height, 0, GL_RGBA, GL_FLOAT, NULL);
	glBindTexture(GL_TEXTURE_2D, 0);
}