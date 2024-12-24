#include "sample_app.hpp"

#include "glm/glm.hpp"
#include "glm/gtc/matrix_transform.hpp"
#include "glm/gtc/quaternion.hpp"
#define GLM_ENABLE_EXPERIMENTAL
#include "glm/gtx/quaternion.hpp"

namespace SampleApp
{
	void SampleAppWindow::onCreate()
	{
		m_viewport_texture.init(m_window_width, m_window_height);
		m_viewport.init(m_viewport_texture);
		m_renderer.init();
		m_application_data.editable_material;
		m_renderer.getMaterial(m_application_data.editable_material_idx,
			&m_application_data.editable_material.albedo,
			&m_application_data.editable_material.metallicity,
			&m_application_data.editable_material.roughness,
			&m_application_data.editable_material.transmission,
			&m_application_data.editable_material.ior
		);

		m_application_data.materials_count = m_renderer.getMaterialsCount();

		m_event_dispatcher.registerListener(Event("exposure_changed"),
			Listener([this](const std::any& data)
				{
					m_camera.setExposure(std::any_cast<float>(data));
					m_renderer.setExposure(m_camera.getExposure());
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
		m_developer_window.draw(m_window_ctx_handle, "Developer Menu");
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