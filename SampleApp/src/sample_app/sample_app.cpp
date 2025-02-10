#include "sample_app.hpp"
#include "imgui_themes.hpp"
#include "model_importer.hpp"

//#define STB_IMAGE_IMPLEMENTATION
//#include "stb/stb_image.h"

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
		std::printf("[APP] initializing app\n");
		m_viewport_texture.init(m_window_width, m_window_height);
		m_viewport.init(m_viewport_texture);
		m_renderer.initialize();

		g_custom_font = ImGuiThemes::VictorSix();
		//ImGuiThemes::Dark();
		//ImGui::StyleColorsDark();

		//loadSceneFile("two_objects.glb");
		loadSceneFile("temp.glb");
		//loadSceneFile("cs16_italy.glb");
		//loadSceneFile("mc_fort.glb");
		//TODO: not robust to empty scenes
		m_application_data.environment_settings = m_renderer.getProceduralEnvironmentData();
		m_application_data.renderer_settings = m_renderer.getRendererSettings();
		//----
		m_application_data.editable_material = m_renderer.getMaterial(m_application_data.editable_material_idx);
		m_application_data.materials_count = m_renderer.getMaterialsCount();
		//----
		m_application_data.editable_mesh_object = m_meshes[m_application_data.editable_mesh_idx];
		m_application_data.meshes_count = m_renderer.getMeshCount();

		KittlesPT::ExposureValues camera_values(m_camera.getApertureF(), m_camera.getISO(), m_camera.getShutterSecs(),
			CameraController::ISO_MAX, CameraController::ISO_MIN,
			1.0f / CameraController::SHUTTER_DENOM_MIN, 1.0f / CameraController::SHUTTER_DENOM_MAX);
		//TODO: have to set these at start; not intuitive
		m_renderer.setExposure(camera_values, m_camera.getExposureCompensationEV(),
			m_camera.getWhitePointEV(), m_camera.getBlackPointEV());

		//TODO: make a constructor/init() for event dispatcher, which registers all listeners, pass it to dev_win
		registerListeners();

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
		m_renderer.resizeResolution(m_window_width, m_window_height);

		//sync client camera with renderer;Assuming aperture priority
		if (m_renderer.getRendererSettings().tonemapper_enable_auto_exposure)
		{
			KittlesPT::ExposureValues cam_val = m_renderer.getExposure();
			m_camera.setISO(static_cast<int>(cam_val.ISO));
			m_camera.setShutterSecs(cam_val.shutter_speed_secs);
		}

		m_renderer.executeRendering(m_developer_window.getDeltaTS_ms());
		//m_renderer.getDebugRenderTargetTexture(m_viewport_texture.m_GL_texture_name);
		m_renderer.getRenderTargetTexture(m_viewport_texture.m_GL_texture_name);

		m_viewport.draw(m_window_ctx_handle);

		if (g_custom_font)ImGui::PushFont(g_custom_font);
		m_developer_window.draw(m_window_ctx_handle, "Sample App Developer Menu");
		if (g_custom_font)ImGui::PopFont();
	}

	void SampleAppWindow::updateUI()
	{
		m_developer_window.updateUI();

		bool view_updated = m_camera.processInput(m_window_ctx_handle, m_developer_window.getDeltaTS_ms());

		if (view_updated)
		{
			glm::mat4 view = m_camera.getViewMatrix();
			glm::mat4 projection = glm::perspectiveFovLH(m_camera.getVerticalFOV_Radians(),
				static_cast<float>(m_window_width), static_cast<float>(m_window_height), 1.0f, 100.0f);
			m_renderer.setView(projection, view);
		}
	}

	//scene parsing
	void SampleAppWindow::loadSceneFile(const char* file_path)
	{
		KittlesPT::BasicScene scene;

		ModelImporter importer;
		importer.loadGLTFfromFile(file_path, &scene);

		for (int32_t i = 0; i < scene.mesh_entities.size(); i++) {
			m_meshes.push_back(MeshObject(i, scene.mesh_entities[i].model_matrix));
		}

		if (!scene.camera_entities.empty()) {
			const KittlesPT::CameraSceneEntity& parsed_camera = scene.camera_entities[0];
			m_camera.setVerticalFOV_Radians(parsed_camera.y_fov_radians);
			auto model = glm::inverse(parsed_camera.view_matrix);
			m_camera.setPosition(model[3]);
			m_camera.setLookAt(-model[2]);
		}

		m_renderer.loadScene(scene);
	}

	//register event listeners
	void SampleAppWindow::registerListeners()
	{
		m_event_dispatcher.registerListener(Event("exposure_changed"),
			Listener([this](const std::any& data)
				{
					std::vector<float>cm_val = std::any_cast<std::vector<float>>(data);

					m_camera.setAperture(cm_val[0]);
					m_camera.setExposureCompensationEV(cm_val[1]);
					m_camera.setISO(static_cast<int>(cm_val[2]));
					m_camera.setShutterSecs(cm_val[3]);
					m_camera.setWhitePointEV(cm_val[4]);
					m_camera.setBlackPointEV(cm_val[5]);

					KittlesPT::ExposureValues camera_values(m_camera.getApertureF(), m_camera.getISO(), m_camera.getShutterSecs(),
						CameraController::ISO_MAX, CameraController::ISO_MIN,
						1.0f / CameraController::SHUTTER_DENOM_MIN, 1.0f / CameraController::SHUTTER_DENOM_MAX);

					m_renderer.setExposure(camera_values, m_camera.getExposureCompensationEV(),
						m_camera.getWhitePointEV(), m_camera.getBlackPointEV());
				}));

		m_event_dispatcher.registerListener(Event("material_changed"),
			Listener([this](const std::any& data)
				{
					m_application_data.editable_material = m_renderer.getMaterial(m_application_data.editable_material_idx);
				}));

		m_event_dispatcher.registerListener(Event("material_updated"),
			Listener([this](const std::any& data)
				{
					m_application_data.editable_material = std::any_cast<KittlesPT::MaterialSceneEntity>(data);

					m_renderer.setMaterial(m_application_data.editable_material_idx,
						m_application_data.editable_material);
				}));

		m_event_dispatcher.registerListener(Event("mesh_changed"),
			Listener([this](const std::any& data)
				{
					m_application_data.editable_mesh_object = m_meshes[m_application_data.editable_mesh_idx];
				}));

		m_event_dispatcher.registerListener(Event("mesh_updated"),
			Listener([this](const std::any& data)
				{
					m_application_data.editable_mesh_object = std::any_cast<MeshObject>(data);

					m_meshes[m_application_data.editable_mesh_idx] = m_application_data.editable_mesh_object;
					m_meshes[m_application_data.editable_mesh_idx].updateMeshTransform(m_renderer);
				}));

		m_event_dispatcher.registerListener(Event("fov_changed"),
			Listener([this](const std::any& data)
				{
					m_camera.setVerticalFOV_Radians(std::any_cast<float>(data));
					glm::mat4 view = m_camera.getViewMatrix();
					glm::mat4 proj = glm::perspectiveFovLH(m_camera.getVerticalFOV_Radians(),
						static_cast<float>(m_window_width), static_cast<float>(m_window_height), 1.0f, 100.0f);
					m_renderer.setView(proj, view);
				}));

		m_event_dispatcher.registerListener(Event("movement_speed_changed"),
			Listener([this](const std::any& data)
				{
					m_camera.setMovementSpeed(std::any_cast<float>(data));
				}));

		m_event_dispatcher.registerListener(Event("pathtracer_settings_changed"),
			Listener([this](const std::any& data)
				{
					m_application_data.renderer_settings = std::any_cast<KittlesPT::RendererSettings>(data);
					m_renderer.setRendererSettings(m_application_data.renderer_settings);
				}));

		m_event_dispatcher.registerListener(Event("environment_settings_changed"),
			Listener([this](const std::any& data)
				{
					m_application_data.environment_settings = std::any_cast<KittlesPT::ProceduralEnvironmentSettings>(data);
					m_renderer.setProceduralEnvironmentData(m_application_data.environment_settings);
				}));
	}
}