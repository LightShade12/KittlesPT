#pragma once
#include "glad/glad.h"
#include "glm/glm.hpp"

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

		float getMaterialRoughness(int idx);

		void getRenderTargetTexture(GLuint r_texture);

		//TODO:make const&
		void setView(glm::mat4 projection_mat, glm::mat4 view_mat);
		/*void loadScene() {};
		void loadSettings() {};*/

		int m_width = 0, m_height = 0;

		RendererData* m_renderer_data = nullptr;
	};
}