#pragma once

#include "pod_types.hpp"

#include "glad/glad.h"
#include "glm/glm.hpp"

namespace KittlesPT
{
	struct RendererResource;

	class Renderer
	{
	public:

		struct ExposureValues
		{
			ExposureValues() = default;

			ExposureValues(float aperture_f_num, int iso, float shutter_secs,
				int max_iso = 6400, int min_iso = 100, float max_shutter_secs = 0.0333f, float min_shutter_secs = 0.00025f) :
				aperture_f_num(aperture_f_num), ISO((float)iso), shutter_speed_secs(shutter_secs),
				MAX_ISO(max_iso), MIN_ISO(min_iso), MAX_SHUTTER_SECS(max_shutter_secs), MIN_SHUTTER_SECS(min_shutter_secs)
			{}

			float aperture_f_num = 0.0f;
			float ISO = 0.0f;
			float shutter_speed_secs = 0.0f;

			int MAX_ISO = 6400;
			int MIN_ISO = 100;
			float MIN_SHUTTER_SECS = 0.00025f;//1/4000s
			float MAX_SHUTTER_SECS = 0.0333f;//1/30s
		};

		void init();

		void shutdown();

		void resizeFrame(int width, int height);

		void executeRendering(float delta_time_ms);

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

		void setExposure(ExposureValues camera_values, float ev_comp, float white_point_ev, float black_point_ev);
		ExposureValues getExposure();

		void resetAccumulation();

		//TODO:make const&
		void setView(glm::mat4 projection_mat, glm::mat4 view_mat);

		void loadScene(const BasicScene& parsed_scene);
		/*void loadSettings() {};*/

	private:
		void submitScene();
		void executeBloomGeneration();
		int m_width = 0, m_height = 0;

		RendererResource* m_renderer_rsrc = nullptr;
	};
}