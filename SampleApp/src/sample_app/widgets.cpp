#include "widgets.hpp"

#include "shared_state.hpp"
#include "glad/include/glad/glad.h"
#define GLFW_INCLUDE_NONE //glad loader instead of local gl
#include "glfw/include/GLFW/glfw3.h"
#include "glm/glm.hpp"

//===========================================================================================
//BACKGROUND TEXTURE
//===========================================================================================
namespace SampleApp
{
	void BackgroundTexture::init(GLTexture bg_texture)
	{
		m_background_texture = bg_texture;
	}

	void BackgroundTexture::draw(GLFWwindow* window_ctx)
	{
		int winposx, winposy;
		int win_width, win_height;
		glfwGetWindowPos(window_ctx, &winposx, &winposy);
		glfwGetFramebufferSize(window_ctx, &win_width, &win_height);
		ImGui::SetNextWindowPos(ImVec2(winposx, winposy));
		ImGui::SetNextWindowSize(ImVec2(win_width, win_height));

		ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.20f, 0.20f, 0.20f, 1.0f));
		ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
		ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
		ImGui::Begin("###viewport", nullptr,
			ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize |
			ImGuiWindowFlags_NoDocking | ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);
		if (m_background_texture.isValid())
		{
			ImGui::Image((void*)m_background_texture.m_GL_texture_name,
				ImVec2(win_width, win_height), { 0,1 }, { 1,0 });
		}

		ImGui::End();
		ImGui::PopStyleVar(2);
		ImGui::PopStyleColor(1);
	};

	//===========================================================================================
	//DEVELOPER WINDOW
	//===========================================================================================

	void DeveloperWindow::updateUI()
	{
		std::chrono::time_point<std::chrono::steady_clock> current_frame_time_point = std::chrono::high_resolution_clock::now();
		delta_time_secs = current_frame_time_point - last_frame_time_point;

		last_frame_time_point = current_frame_time_point;
	}

	void DeveloperWindow::renderUI()
	{
		ImGui::Text("Delta ms(last frame): %.3f ms", delta_time_secs.count() * 1000.0f);
		float fps = 1000.0f / (delta_time_secs.count() * 1000.0f);
		ImGui::Text("FPS(last frame): %.3f ms", fps);

		//TODO: weird; idk
		float avg = glm::mix(average_fps, fps, 0.01f);
		if (!isnan(avg) && !isinf(avg))
		{
			average_fps = avg;
		}

		ImGui::Text("EMA FPS: %.3f ms", average_fps);
		ImGui::Text("Runtime secs: %.3f s",
			std::chrono::duration_cast<std::chrono::duration<float>>(last_frame_time_point.time_since_epoch()).count() - start_time_secs);
		ImGui::Separator();

		ImGui::SeparatorText("Camera edit");
		if (camera_controller_ref != nullptr)
		{
			float fov_y_rad = camera_controller_ref->getVerticalFOV_Radians();
			float exposure = camera_controller_ref->getExposure();
			if (ImGui::SliderAngle("FOV", &fov_y_rad, 0.0, 120.0))
			{
				event_dispatcher_ref->emitSignal(Event("fov_changed"), fov_y_rad);
			};
			if (ImGui::SliderFloat("Exposure", &exposure, 0.0, 100.0,
				"%.3f unitless", ImGuiSliderFlags_Logarithmic)) {
				event_dispatcher_ref->emitSignal(Event("exposure_changed"), exposure);
			};
		}
		ImGui::SeparatorText("Material edit");
		bool material_updated = false;
		if (ImGui::SliderInt("Material selection:", &shared_data_ref->editable_material_idx,
			0, shared_data_ref->materials_count - 1))
		{
			event_dispatcher_ref->emitSignal(Event("material_changed"), true);
		};

		Material material = shared_data_ref->editable_material;
		material_updated |= ImGui::ColorEdit3("Albedo factor", &material.albedo.r);
		material_updated |= ImGui::SliderFloat("Metallicity", &material.metallicity, 0, 1);
		material_updated |= ImGui::SliderFloat("Isotropic roughness", &material.roughness, 0, 1);
		material_updated |= ImGui::SliderFloat("Transmission", &material.transmission, 0, 1);
		material_updated |= ImGui::SliderFloat("IOR", &material.ior, 0, 2);
		if (material_updated)
		{
			event_dispatcher_ref->emitSignal(Event("material_updated"), material);
		}
	}
}