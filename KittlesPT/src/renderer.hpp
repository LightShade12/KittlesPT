#pragma once
#include "glad/glad.h"

namespace KittlesPT
{
	struct RendererData;

	class Renderer
	{
	public:

		void init();

		void resizeFrame(int width, int height);

		void shutdown();

		void executeRendering();

		void getRenderTargetTexture(GLuint r_texture);

		/*void loadScene() {};
		void loadSettings() {};
		void setView() {};*/

		int m_width = 0, m_height = 0;

		RendererData* m_renderer_data = nullptr;
	};
}