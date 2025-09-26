#pragma once

#include "pod_types.hpp"

#include "glad/glad.h"
#include "glm/glm.hpp"

//TODO: a more verbose name for Renderer
namespace KittlesPT
{
	struct RendererResource;

	class Renderer
	{
	public:

		void initialize();
		void shutdown();

		void resizeResolution(uint32_t width, uint32_t height);

		void executeRendering(float delta_time_ms);

		void getRenderTargetTexture(GLuint r_texture) const;
		void getDebugRenderTargetTexture(GLuint r_texture) const;

		size_t getMaterialsCount() const;
		size_t getMeshCount() const;

		void loadScene(const BasicScene& parsed_scene);

		void setExposure(const ExposureValues& camera_values, float ev_comp, float white_point_ev, float black_point_ev);
		ExposureValues getExposure() const;

		bool setMaterial(uint32_t index, const MaterialSceneEntity& material);
		MaterialSceneEntity getMaterial(uint32_t index) const;

		bool setMeshTransform(uint32_t index, const glm::mat4& model_matrix);
		glm::mat4 getMeshTransform(uint32_t index) const;

		void setProceduralEnvironmentData(const ProceduralEnvironmentSettings& settings);
		ProceduralEnvironmentSettings getProceduralEnvironmentData() const;

		void setRendererSettings(const RendererSettings& settings);
		RendererSettings getRendererSettings() const;

		void setView(const glm::mat4& projection_matrix, const glm::mat4& view_matrix);

		void resetAccumulation();

	private:
		void executeBloomGeneration();

		uint32_t m_output_width = 0, m_output_height = 0;
		uint32_t m_render_width = 0, m_render_height = 0;

		RendererResource* m_renderer_rsrc = nullptr;
	};
}/*KittlesPT*/