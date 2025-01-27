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
		m_renderer.init();

		//g_custom_font = ImGuiThemes::VictorSix();
		//ImGuiThemes::Dark();
		ImGui::StyleColorsDark();

		loadSceneFile("test.glb");

		m_application_data.editable_material = m_renderer.getMaterial(m_application_data.editable_material_idx);

		m_application_data.materials_count = m_renderer.getMaterialsCount();
		m_application_data.environment_data = m_renderer.getProceduralEnvironmentData();
		m_application_data.renderer_settings = m_renderer.getRendererSettings();

		KittlesPT::Renderer::ExposureValues camera_values(m_camera.getAperture(), m_camera.getISO(), m_camera.getShutterSecs(),
			CameraController::ISO_MAX, CameraController::ISO_MIN,
			1.0f / CameraController::SHUTTER_DENOM_MIN, 1.0f / CameraController::SHUTTER_DENOM_MAX);

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
		m_renderer.resizeFrame(m_window_width, m_window_height);

		if (m_renderer.getRendererSettings().enable_auto_exposure)
		{
			KittlesPT::Renderer::ExposureValues cam_val = m_renderer.getExposure();
			m_camera.setISO(static_cast<int>(cam_val.ISO));
			m_camera.setShutterSecs(cam_val.shutter_speed_secs);
		}

		m_renderer.executeRendering(m_developer_window.getDeltaTS_ms());
		m_renderer.getRenderTargetTexture(m_viewport_texture.m_GL_texture_name);

		m_viewport.draw(m_window_ctx_handle);

		if (g_custom_font)ImGui::PushFont(g_custom_font);
		m_developer_window.draw(m_window_ctx_handle, "Developer Menu");
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

		////Image load
		//int width = 0, height = 0, channels = 0;
		//unsigned char* img_data = stbi_load("grid.png", &width, &height, &channels, 3);
		//KittlesPT::TextureSceneEntity texture0(img_data, width, height, 3);
		//stbi_image_free(img_data);
		//img_data = nullptr;
		//scene.addTexture(texture0);

		//scene.addMaterial(KittlesPT::MaterialSceneEntity(
		//	0, glm::vec3(1.0, 1.0, 1.0),
		//	-1, 0.0f, 0.8f,
		//	-1, 0.0f,
		//	1.45f,
		//	-1, glm::vec3(0.0f), 1.0f,
		//	-1, 1.0f
		//));
		//{
		//	float side_length = 2.0f;  // Length of one side of the square

		//	glm::vec3 p0(-side_length / 2.0f, side_length / 2.0f, 0.0f);  // Top-left
		//	glm::vec3 p1(-side_length / 2.0f, -side_length / 2.0f, 0.0f); // Bottom-left
		//	glm::vec3 p2(side_length / 2.0f, -side_length / 2.0f, 0.0f);  // Bottom-right
		//	glm::vec3 p3(side_length / 2.0f, side_length / 2.0f, 0.0f);   // Top-right

		//	glm::vec3 n0(0.0f, 0.0f, 1.0f);
		//	glm::vec3 n1(0.0f, 0.0f, 1.0f);
		//	glm::vec3 n2(0.0f, 0.0f, 1.0f);
		//	glm::vec3 n3(0.0f, 0.0f, 1.0f);

		//	glm::vec2 t0(0.0f, 1.0f);  // Top-left
		//	glm::vec2 t1(0.0f, 0.0f);  // Bottom-left
		//	glm::vec2 t2(1.0f, 0.0f);  // Bottom-right
		//	glm::vec2 t3(1.0f, 1.0f);  // Top-right

		//	int material_id = 0;

		//	// Create two triangle instances for the square
		//	KittlesPT::TriangleSceneEntity triangle1(p0, p1, p2, n0, n1, n2, t0, t1, t2, material_id);  // First triangle (lower)
		//	KittlesPT::TriangleSceneEntity triangle2(p0, p2, p3, n0, n2, n3, t0, t2, t3, material_id);  // Second triangle (upper)

		//	scene.addShape(triangle1);
		//	scene.addShape(triangle2);
		//}

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

					KittlesPT::Renderer::ExposureValues camera_values(m_camera.getAperture(), m_camera.getISO(), m_camera.getShutterSecs(),
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

		m_event_dispatcher.registerListener(Event("fov_changed"),
			Listener([this](const std::any& data)
				{
					m_camera.setVerticalFOV_Radians(std::any_cast<float>(data));
					glm::mat4 view = m_camera.getViewMatrix();
					glm::mat4 proj = glm::perspectiveFovLH(m_camera.getVerticalFOV_Radians(),
						static_cast<float>(m_window_width), static_cast<float>(m_window_height), 1.0f, 100.0f);
					m_renderer.setView(proj, view);
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
					m_application_data.environment_data = std::any_cast<KittlesPT::ProceduralEnvironmentData>(data);
					m_renderer.setProceduralEnvironmentData(m_application_data.environment_data);
				}));
	}
}