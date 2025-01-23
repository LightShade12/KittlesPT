#include "widgets.hpp"

#include "shared_state.hpp"
#include "glad/include/glad/glad.h"
#define GLFW_INCLUDE_NONE //glad loader instead of local gl
#include "glfw/include/GLFW/glfw3.h"
#include "glm/glm.hpp"
#include <vector>

namespace SampleApp
{
	//===========================================================================================
	//BACKGROUND TEXTURE
	//===========================================================================================

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
		ImGui::SetNextWindowPos(ImVec2((float)winposx, (float)winposy));
		ImGui::SetNextWindowSize(ImVec2((float)win_width, (float)win_height));

		ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.20f, 0.20f, 0.20f, 1.0f));
		ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
		ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
		ImGui::Begin("###viewport", nullptr,
			ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize |
			ImGuiWindowFlags_NoDocking | ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);
		if (m_background_texture.isValid())
		{
			ImGui::Image((ImTextureID*)(long long(m_background_texture.m_GL_texture_name)),
				ImVec2((float)win_width, (float)win_height), { 0,1 }, { 1,0 });
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
		ImGui::Text("FPS(last frame): %.3f", fps);

		//TODO: weird; idk
		float avg = glm::mix(average_fps, fps, 0.01f);
		if (!isnan(avg) && !isinf(avg))
		{
			average_fps = avg;
		}

		ImGui::Text("EMA FPS: %.3f", average_fps);
		ImGui::Text("Runtime secs: %.3f s",
			std::chrono::duration_cast<std::chrono::duration<float>>(last_frame_time_point.time_since_epoch()).count() - start_time_secs);

		//ImGui::ShowDemoWindow();

		//camera edit
		ImGui::Separator();
		if (ImGui::CollapsingHeader("Camera Settings") && camera_controller_ref != nullptr) {
			float fov_y_radians = camera_controller_ref->getVerticalFOV_Radians();

			float aperture_f_num = camera_controller_ref->getAperture();
			float exp_comp = camera_controller_ref->getExposureCompensationEV();
			int ISO = camera_controller_ref->getISO();
			float shutter_secs = camera_controller_ref->getShutterSecs();

			//TODO: make this into CameraSettings struct?
			std::vector<float>camera_values = { aperture_f_num,exp_comp,static_cast<float>(ISO),shutter_secs,
				camera_controller_ref->getWhitePointEV(),camera_controller_ref->getBlackPointEV() };

			bool exposure_updated = false;

			if (ImGui::BeginTable("cameraedittable", 2))
			{
				ImGui::TableSetupColumn("A0", 0, 0.4f);
				ImGui::TableSetupColumn("A1", 0);

				ImGui::TableNextRow();
				ImGui::TableSetColumnIndex(0);
				ImGui::Text("Vertical FOV");
				ImGui::TableSetColumnIndex(1);
				ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x);
				if (ImGui::SliderAngle("###fov_control", &fov_y_radians, 0.0, 120.0)) {
					event_dispatcher_ref->emitSignal(Event("fov_changed"), fov_y_radians);
				};

				ImGui::TableNextRow();
				ImGui::TableSetColumnIndex(0);
				ImGui::Text("EV Compensation");
				ImGui::TableSetColumnIndex(1);
				ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x);
				exposure_updated |= ImGui::SliderFloat("###evcomp_control", &camera_values[1],
					-3.0f, 3.0, "%.3f EV");

				ImGui::TableNextRow();
				ImGui::TableSetColumnIndex(0);
				ImGui::Text("Aperture F-Stops");
				ImGui::TableSetColumnIndex(1);
				ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x);
				exposure_updated |= ImGui::SliderFloat("###aperture_control", &camera_values[0],
					CameraController::AP_F_MIN, CameraController::AP_F_MAX, "f/%.3f");

				ImGui::TableNextRow();
				ImGui::TableSetColumnIndex(0);
				ImGui::Text("ISO");
				ImGui::TableSetColumnIndex(1);
				ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x);
				int factor = static_cast<int>(log2(ISO / 100));//keep it outside; mind the slider modifying ISO below
				if (ImGui::InputInt("###iso_control", &ISO, 100, 100, ImGuiInputTextFlags_EnterReturnsTrue)) {
					bool inc = (camera_values[2] < ISO);
					factor = glm::max(factor - (!inc), 0);
					ISO = static_cast<int>(camera_values[2]) + (((inc) ? 1 : -1) * 100 * static_cast<int>(pow(2, factor)));
					camera_values[2] = static_cast<float>(glm::clamp(ISO,
						CameraController::ISO_MIN, CameraController::ISO_MAX));
					exposure_updated |= true;
				};

				ImGui::TableNextRow();
				ImGui::TableSetColumnIndex(0);
				ImGui::Text("Shutter Speed");
				ImGui::TableSetColumnIndex(1);
				ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x);
				float denominator = 1.0f / shutter_secs;
				if (ImGui::SliderFloat("###shutter_control", &denominator,
					CameraController::SHUTTER_DENOM_MIN, CameraController::SHUTTER_DENOM_MAX, "1/%.1fs",
					ImGuiSliderFlags_Logarithmic)) {
					shutter_secs = 1.0f / denominator;
					camera_values[3] = shutter_secs;
					exposure_updated |= true;
				};

				ImGui::EndTable();
			}
			ImGui::SeparatorText("View Transform");
			if (ImGui::BeginTable("cameracolortransformedittable", 2))
			{
				ImGui::TableSetupColumn("A0", 0, 0.4f);
				ImGui::TableSetupColumn("A1", 0);

				ImGui::TableNextRow();
				ImGui::TableSetColumnIndex(0);
				ImGui::Text("Dynamic Range:");
				ImGui::TableSetColumnIndex(1);
				ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x);
				ImGui::Text("%.3f EV", camera_values[4] - camera_values[5]);

				ImGui::TableNextRow();
				ImGui::TableSetColumnIndex(0);
				ImGui::Text("White Point");
				ImGui::TableSetColumnIndex(1);
				ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x);
				exposure_updated |= ImGui::SliderFloat("###white_point_control", &camera_values[4],
					5.0f, 20.0f, "%.3f EV");

				ImGui::TableNextRow();
				ImGui::TableSetColumnIndex(0);
				ImGui::Text("Black Point");
				ImGui::TableSetColumnIndex(1);
				ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x);
				exposure_updated |= ImGui::SliderFloat("###black_point_control", &camera_values[5],
					-15.0f, -3.0f, "%.3f EV");

				ImGui::EndTable();
			}

			if (exposure_updated)
			{
				event_dispatcher_ref->emitSignal(Event("exposure_changed"), camera_values);
			};
		};

		//integrator edit
		ImGui::Separator();
		if (ImGui::CollapsingHeader("Integrator Edit")) {
			KittlesPT::PathtracerSettings pt_settings = shared_data_ref->pathtracer_settings;
			bool pt_settings_updated = false;

			if (ImGui::BeginTable("integratoredittable", 2)) {
				ImGui::TableSetupColumn("A0", 0, 0.4f);
				ImGui::TableSetupColumn("A1", 0);

				ImGui::TableNextRow();
				ImGui::TableSetColumnIndex(0);
				ImGui::Text("Max Path Depth");
				ImGui::TableSetColumnIndex(1);
				ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x);
				pt_settings_updated |= ImGui::SliderInt("###max_bounces", &pt_settings.max_bounce_depth, 0, 32);
				ImGui::EndTable();

				pt_settings_updated |= ImGui::Checkbox("Enable AutoExposure", &pt_settings.enable_auto_exposure);

				pt_settings_updated |= ImGui::Checkbox("Generate Veiling Luminance(Bloom)", &pt_settings.generate_bloom);
				ImGui::Indent();
				pt_settings_updated |= ImGui::Checkbox("Use Karis Average", &pt_settings.use_karis_average);
				if (ImGui::BeginTable("bloomedittable", 2)) {
					ImGui::TableSetupColumn("A0", 0, 0.8f);
					ImGui::TableSetupColumn("A1", 0);

					ImGui::TableNextRow();
					ImGui::TableSetColumnIndex(0);
					ImGui::Text("Bloom Blend Factor");
					ImGui::TableSetColumnIndex(1);
					ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x);
					pt_settings_updated |= ImGui::SliderFloat("###bloomblendfac", &pt_settings.bloom_blend, 0.0f, 1.0f);

					ImGui::TableNextRow();
					ImGui::TableSetColumnIndex(0);
					ImGui::Text("Bloom Internal Blend Factor");
					ImGui::TableSetColumnIndex(1);
					ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x);
					pt_settings_updated |= ImGui::SliderFloat("###bloomintblendfac", &pt_settings.bloom_internal_blend, 0.0f, 1.0f);

					ImGui::EndTable();
				}
				ImGui::Unindent();
			}

			if (pt_settings_updated) {
				event_dispatcher_ref->emitSignal(Event("pathtracer_settings_changed"), pt_settings);
			}
		}

		//environment edit
		ImGui::Separator();
		if (ImGui::CollapsingHeader("Environment Edit"))
		{
			KittlesPT::ProceduralEnvironmentData env_data = shared_data_ref->environment_data;
			bool env_updated = false;

			if (ImGui::BeginTable("envedittable", 2))
			{
				ImGui::TableSetupColumn("A0", 0, 0.6f);
				ImGui::TableSetupColumn("A1", 0);

				ImGui::TableNextRow();
				ImGui::TableSetColumnIndex(0);
				ImGui::Text("Sun Radiance");
				ImGui::TableSetColumnIndex(1);
				ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x);
				env_updated |= ImGui::SliderFloat("###sun_radiance", &env_data.sun_emission_nits, 0, 6e5f,
					"%.3f nits", ImGuiSliderFlags_Logarithmic);

				ImGui::TableNextRow();
				ImGui::TableSetColumnIndex(0);
				ImGui::Text("Sun Angular Diameter");
				ImGui::TableSetColumnIndex(1);
				ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x);
				env_updated |= ImGui::SliderAngle("###sun_angular_diameter", &env_data.sun_angular_diameter_rad, 0, 45,
					"%.1f deg", ImGuiSliderFlags_Logarithmic);

				ImGui::TableNextRow();
				ImGui::TableSetColumnIndex(0);
				ImGui::Text("Sun Altitude Theta");
				ImGui::TableSetColumnIndex(1);
				ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x);
				env_updated |= ImGui::SliderAngle("###sun_altitude_theta", &env_data.sun_theta_rad, -20, 90);

				ImGui::TableNextRow();
				ImGui::TableSetColumnIndex(0);
				ImGui::Text("Sun Position Phi");
				ImGui::TableSetColumnIndex(1);
				ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x);
				env_updated |= ImGui::SliderAngle("###sun_phi", &env_data.sun_phi_rad, 0, 360);

				ImGui::EndTable();
			}

			if (env_updated) {
				event_dispatcher_ref->emitSignal(Event("environment_settings_changed"), env_data);
			}
		}

		//material edit
		ImGui::Separator();
		if (ImGui::CollapsingHeader("Material Edit"))
		{
			bool material_updated = false;
			Material material;

			if (ImGui::BeginTable("materialedittable", 2))
			{
				ImGui::TableSetupColumn("A0", 0, 0.6f);
				ImGui::TableSetupColumn("A1", 0);

				ImGui::TableNextRow();
				ImGui::TableSetColumnIndex(0);
				ImGui::Text("Material ID:");
				ImGui::TableSetColumnIndex(1);
				ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x);
				if (ImGui::SliderInt("###material_selection", &shared_data_ref->editable_material_idx,
					0, shared_data_ref->materials_count - 1))
				{
					event_dispatcher_ref->emitSignal(Event("material_changed"), true);
				};

				material = shared_data_ref->editable_material;

				ImGui::TableNextRow();
				ImGui::TableSetColumnIndex(0);
				ImGui::Text("Albedo Factor");
				ImGui::TableSetColumnIndex(1);
				ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x);
				material_updated |= ImGui::ColorEdit3("###albedo", &material.albedo.r);

				ImGui::TableNextRow();
				ImGui::TableSetColumnIndex(0);
				ImGui::Text("Metallness Factor");
				ImGui::TableSetColumnIndex(1);
				ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x);
				material_updated |= ImGui::SliderFloat("###metallicity", &material.metallicity, 0.0f, 1.0f);

				ImGui::TableNextRow();
				ImGui::TableSetColumnIndex(0);
				ImGui::Text("Roughness Factor");
				ImGui::TableSetColumnIndex(1);
				ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x);
				material_updated |= ImGui::SliderFloat("###roughness", &material.roughness, 0.0f, 1.0f);

				ImGui::TableNextRow();
				ImGui::TableSetColumnIndex(0);
				ImGui::Text("Transmission Factor");
				ImGui::TableSetColumnIndex(1);
				ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x);
				material_updated |= ImGui::SliderFloat("###transmission", &material.transmission, 0, 1);

				ImGui::TableNextRow();
				ImGui::TableSetColumnIndex(0);
				ImGui::Text("IOR Factor");
				ImGui::TableSetColumnIndex(1);
				ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x);
				material_updated |= ImGui::SliderFloat("###ior", &material.ior, 1, 3);

				ImGui::EndTable();
			}

			if (material_updated)
			{
				event_dispatcher_ref->emitSignal(Event("material_updated"), material);
			}
		}
	}
}