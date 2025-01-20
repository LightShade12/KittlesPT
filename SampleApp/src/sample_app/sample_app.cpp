#include "sample_app.hpp"
#include "imgui_themes.hpp"

#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"

#include "glm/glm.hpp"
#include "glm/gtc/matrix_transform.hpp"
#include "glm/gtc/quaternion.hpp"
#define GLM_ENABLE_EXPERIMENTAL
#include "glm/gtx/quaternion.hpp"

namespace SampleApp
{
	ImFont* g_custom_font = nullptr;

	void SampleAppWindow::onCreate()
	{
		m_viewport_texture.init(m_window_width, m_window_height);
		m_viewport.init(m_viewport_texture);
		m_renderer.init();

		//g_custom_font = ImGuiThemes::VictorSix();
		//ImGuiThemes::Dark();
		ImGui::StyleColorsDark();

		//scene parsing
		{
			KittlesPT::BasicScene scene;

			//Image load
			int width = 0, height = 0, channels = 0;

			unsigned char* img_data = stbi_load("blackpaint.png", &width, &height, &channels, 3);
			KittlesPT::TextureSceneEntity texture0(img_data, width, height, 3);
			stbi_image_free(img_data);
			img_data = nullptr;

			scene.addTexture(texture0);

			img_data = stbi_load("grid.png", &width, &height, &channels, 3);
			KittlesPT::TextureSceneEntity texture1(img_data, width, height, 3);
			stbi_image_free(img_data);
			img_data = nullptr;

			scene.addTexture(texture1);

			img_data = stbi_load("colors2.jpg", &width, &height, &channels, 3);
			KittlesPT::TextureSceneEntity texture2(img_data, width, height, 3);
			stbi_image_free(img_data);
			img_data = nullptr;

			scene.addTexture(texture2);

			img_data = stbi_load("uvgrid.jpeg", &width, &height, &channels, 3);
			KittlesPT::TextureSceneEntity texture3(img_data, width, height, 3);
			stbi_image_free(img_data);
			img_data = nullptr;

			scene.addTexture(texture3);

			scene.addMaterial(KittlesPT::MaterialSceneEntity(
				0, glm::vec3(1.0, 1.0, 1.0),
				-1, 0.0f, 0.1f,
				-1, 0.0f,
				1.45f,
				-1, glm::vec3(0.0f), 1.0f,
				-1, 1.0f
			));

			scene.addMaterial(KittlesPT::MaterialSceneEntity(
				3, glm::vec3(0.5, 0.5, 0.5),
				-1, 0.0f, 0.1f,
				-1, 0.0f,
				1.45f,
				-1, glm::vec3(0.0f), 1.0f,
				-1, 1.0f
			));

			scene.addMaterial(KittlesPT::MaterialSceneEntity(
				2, glm::vec3(0.9f, 0.9f, 0.9f),
				-1, 1.0f, 0.3f,
				-1, 0.0f,
				1.45f,
				-1, glm::vec3(0.0f), 1.0f,
				-1, 1.0f
			));

			scene.addMaterial(KittlesPT::MaterialSceneEntity(
				1, glm::vec3(1.0f, 1.0f, 1.0f),
				-1, 0.0f, 0.0f,
				-1, 1.0f,
				1.45f,
				-1, glm::vec3(0.0f), 1.0f,
				-1, 1.0f
			));

			//60-watt lightbulb emission; 800 lm; 233 nits
			scene.addMaterial(KittlesPT::MaterialSceneEntity(
				-1, glm::vec3(0.0f, 1.0f, 0.0f),
				-1, 0.0f, 0.85f,
				-1, 0.0f,
				1.45f,
				-1, glm::vec3(0.2f, 0.7f, 1.0f), 12.0e4f,
				-1, 1.0f
			));

			scene.addShape(KittlesPT::SphereSceneEntity(0.5f, glm::vec3(0, 0, -3), 0));
			scene.addShape(KittlesPT::SphereSceneEntity(0.5f, glm::vec3(-1.5, 0, -3), 2));
			scene.addShape(KittlesPT::SphereSceneEntity(0.5f, glm::vec3(1.5, 0, -3), 3));
			scene.addShape(KittlesPT::SphereSceneEntity(0.5f, glm::vec3(0, 1.5, -3), 4));//light source
			scene.addShape(KittlesPT::SphereSceneEntity(100.0f, glm::vec3(0, -100.5, -3), 1));//textured

			m_renderer.loadScene(scene);
		}

		m_renderer.getMaterial(m_application_data.editable_material_idx,
			&m_application_data.editable_material.albedo,
			&m_application_data.editable_material.metallicity,
			&m_application_data.editable_material.roughness,
			&m_application_data.editable_material.transmission,
			&m_application_data.editable_material.ior
		);

		m_application_data.materials_count = m_renderer.getMaterialsCount();
		m_application_data.environment_data = m_renderer.getProceduralEnvironmentData();
		m_application_data.pathtracer_settings = m_renderer.getPathTracerSettings();
		m_renderer.setExposure(m_camera.getAperture(), m_camera.getISO(),
			m_camera.getShutter(), m_camera.getExposureCompensation(),
			m_camera.getWhitePoint(), m_camera.getBlackPoint());

		//========================================================================================================
		//REGISTER EVENT LISTENERS
		//========================================================================================================

		m_event_dispatcher.registerListener(Event("exposure_changed"),
			Listener([this](const std::any& data)
				{
					std::vector<float>cm_val = std::any_cast<std::vector<float>>(data);

					m_camera.setAperture(cm_val[0]);
					m_camera.setExposureCompensation(cm_val[1]);
					m_camera.setISO(static_cast<int>(cm_val[2]));
					m_camera.setShutter(cm_val[3]);
					m_camera.setWhitePoint(cm_val[4]);
					m_camera.setBlackPoint(cm_val[5]);

					m_renderer.setExposure(m_camera.getAperture(), m_camera.getISO(),
						m_camera.getShutter(), m_camera.getExposureCompensation(),
						m_camera.getWhitePoint(), m_camera.getBlackPoint());
				}));

		m_event_dispatcher.registerListener(Event("material_changed"),
			Listener([this](const std::any& data)
				{
					m_renderer.getMaterial(m_application_data.editable_material_idx,
						&m_application_data.editable_material.albedo,
						&m_application_data.editable_material.metallicity,
						&m_application_data.editable_material.roughness,
						&m_application_data.editable_material.transmission,
						&m_application_data.editable_material.ior
						);
				}));

		m_event_dispatcher.registerListener(Event("material_updated"),
			Listener([this](const std::any& data)
				{
					m_application_data.editable_material = std::any_cast<Material>(data);
					const Material& mat = m_application_data.editable_material;

					m_renderer.setMaterial(
						m_application_data.editable_material_idx,
						mat.albedo,
						mat.metallicity,
						mat.roughness,
						mat.transmission,
						mat.ior);
				}));

		m_event_dispatcher.registerListener(Event("fov_changed"),
			Listener([this](const std::any& data)
				{
					m_camera.setVerticalFOV_Radians(std::any_cast<float>(data));
					glm::mat4 view = m_camera.getViewMatrix();
					glm::mat4 proj = glm::perspectiveFovLH(m_camera.getVerticalFOV_Radians(),
						float(m_window_width), float(m_window_height), 1.f, 100.f);
					m_renderer.setView(proj, glm::inverse(view));
				}));

		m_event_dispatcher.registerListener(Event("pathtracer_settings_changed"),
			Listener([this](const std::any& data)
				{
					m_application_data.pathtracer_settings = std::any_cast<KittlesPT::PathtracerSettings>(data);
					m_renderer.setPathTracerSettings(m_application_data.pathtracer_settings);
				}));

		m_event_dispatcher.registerListener(Event("environment_settings_changed"),
			Listener([this](const std::any& data)
				{
					m_application_data.environment_data = std::any_cast<KittlesPT::ProceduralEnvironmentData>(data);
					m_renderer.setProceduralEnvironmentData(m_application_data.environment_data);
				}));

		//========================================================================================================

		m_developer_window.init(&m_event_dispatcher, &m_camera,
			&m_application_data);
	}

	void SampleAppWindow::onDestroy()
	{
		m_renderer.shutdown();
		m_viewport_texture.destroy();
	}

	void SampleAppWindow::renderUI()
	{
		m_viewport_texture.resize(m_window_width, m_window_height);
		m_renderer.resizeFrame(m_window_width, m_window_height);
		m_renderer.executeRendering();
		m_renderer.getRenderTargetTexture(m_viewport_texture.m_GL_texture_name);

		m_viewport.draw(m_window_ctx_handle);
		if (g_custom_font)ImGui::PushFont(g_custom_font);
		m_developer_window.draw(m_window_ctx_handle, "Developer Menu");
		if (g_custom_font)ImGui::PopFont();
	}

	void SampleAppWindow::updateUI()
	{
		m_developer_window.updateUI();

		bool view_updated = m_camera.processInput(m_window_ctx_handle, m_developer_window.getDeltaTS());

		if (view_updated)
		{
			glm::mat4 view = m_camera.getViewMatrix();
			glm::mat4 proj = glm::perspectiveFovLH(m_camera.getVerticalFOV_Radians(),
				float(m_window_width), float(m_window_height), 1.f, 100.f);
			m_renderer.setView(proj, glm::inverse(view));
		}
	}
}