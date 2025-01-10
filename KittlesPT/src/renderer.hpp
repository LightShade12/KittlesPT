#pragma once

#include "pod_types.hpp"

#include "glad/glad.h"
#include "glm/glm.hpp"

namespace KittlesPT
{
	struct RendererData;

	class Renderer
	{
	public:

		void init();

		void shutdown();

		void resizeFrame(int width, int height);

		void executeRendering();

		void getRenderTargetTexture(GLuint r_texture);

		void getDebugRenderTargetTexture(GLuint r_texture);

		//TODO: work with MaterialSceneEntity
		bool setMaterial(int idx,
			glm::vec3 albedo_factor,
			float metallicity,
			float roughness,
			float transmission,
			float ior
		);

		//TODO: return MaterialSceneEntity
		bool getMaterial(int idx,
			glm::vec3* albedo_factor,
			float* metallicity,
			float* roughness,
			float* transmission,
			float* ior
		);

		int getMaterialsCount();

		void setProceduralEnvironmentData(ProceduralEnvironmentData data);
		ProceduralEnvironmentData getProceduralEnvironmentData();

		void setPathTracerSettings(PathtracerSettings cfg);
		PathtracerSettings getPathTracerSettings();

		void setExposure(float exposure);

		void resetAccumulation();

		//TODO:make const&
		void setView(glm::mat4 projection_mat, glm::mat4 view_mat);

		void loadScene(const BasicScene& parsed_scene);
		/*void loadSettings() {};*/

	private:
		void submitScene();
		void executeBloomGeneration();
		int m_width = 0, m_height = 0;

		RendererData* m_renderer_data = nullptr;
	};
}