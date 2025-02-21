#include "widgets.hpp"

#include "mesh_object.hpp"
#include "shared_state.hpp"
#include "glad/include/glad/glad.h"
#define GLFW_INCLUDE_NONE //glad loader instead of local gl
#include "glfw/include/GLFW/glfw3.h"
#include "glm/glm.hpp"
#include "glm/gtc/matrix_transform.hpp"
#include "glm/gtc/type_ptr.hpp"

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
			ImGui::Image((ImTextureID*)(int64_t(m_background_texture.getGLTexture())),
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
		m_delta_time_secs = current_frame_time_point - m_last_frame_time_point;

		m_last_frame_time_point = current_frame_time_point;
	}

	void DeveloperWindow::renderUI()
	{
		{
			float curr_fps = 1000.0f / (m_delta_time_secs.count() * 1000.0f);
			float runtime_secs = std::chrono::duration_cast<std::chrono::duration<float>>(
				m_last_frame_time_point.time_since_epoch()).count() - m_start_time_secs;

			if (ImGui::BeginTable("mytable", 2))
			{
				ImGui::TableSetupColumn("A0", 0, 0.8);
				ImGui::TableSetupColumn("A1", 0, 0.3);

				ImGui::TableNextRow();
				ImGui::TableSetColumnIndex(0);
				ImGui::Text("Delta time (last frame): %.3f ms", m_delta_time_secs.count() * 1000.0f);
				ImGui::Text("FPS (last frame): %.3f", curr_fps);

				//TODO: weird; idk
				float avg = glm::mix(m_average_fps, curr_fps, 0.01f);
				if (!isnan(avg) && !isinf(avg)) {
					m_average_fps = avg;
				}

				ImGui::Text("EMA FPS: %.3f", m_average_fps);
				ImGui::Text("Runtime secs: %.3f s", runtime_secs);
				ImGui::TableSetColumnIndex(1);
				//ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x);
				//ImGui::Dummy(ImVec2(10,10)); ImGui::SameLine();
				ImGui::Image((ImTextureID*)(int64_t((*m_textures_handle)["img0"].getGLTexture())),
					ImVec2(ImGui::GetContentRegionAvail().x, ImGui::GetContentRegionAvail().x));

				ImGui::EndTable();
			}

			static float values[90] = {};
			static int values_offset = 0;
			static double refresh_time = 0.0;
			if (refresh_time == 0.0) {
				refresh_time = ImGui::GetTime();
			}
			while (refresh_time < ImGui::GetTime()) // create data at fixed 60 Hz rate
			{
				values[values_offset] = curr_fps;
				values_offset = (values_offset + 1) % IM_ARRAYSIZE(values);
				refresh_time += 1.0f / 60.0f;
			}
			{
				char overlay[32];
				sprintf_s(overlay, "average: %.3f fps", m_average_fps);
				ImGui::PlotLines("###fps_plot", values, IM_ARRAYSIZE(values), values_offset,
					overlay, -1.0f, 121.0f, ImVec2(ImGui::GetContentRegionAvail().x, 80.0f));
			}
		}

		//ImGui::ShowDemoWindow();

		//camera edit
		ImGui::Separator();
		if (ImGui::CollapsingHeader("Camera Settings") && m_camera_handle != nullptr) {
			float fov_y_radians = m_camera_handle->getVerticalFOV_Radians();
			float move_speed = m_camera_handle->getMovementSpeed();
			float aperture_f_num = m_camera_handle->getApertureF();
			float exp_comp = m_camera_handle->getExposureCompensationEV();
			int ISO = m_camera_handle->getISO();
			float shutter_secs = m_camera_handle->getShutterSecs();

			//TODO: make this into CameraSettings struct?
			std::vector<float>camera_values = { aperture_f_num,exp_comp,static_cast<float>(ISO),shutter_secs,
				m_camera_handle->getWhitePointEV(),m_camera_handle->getBlackPointEV() };

			bool exposure_updated = false;

			if (ImGui::BeginTable("cameraedittable", 2))
			{
				ImGui::TableSetupColumn("A0", 0, 0.4f);
				ImGui::TableSetupColumn("A1", 0);

				ImGui::TableNextRow();
				ImGui::TableSetColumnIndex(0);
				ImGui::Text("Movement Speed");
				ImGui::TableSetColumnIndex(1);
				ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x);
				if (ImGui::SliderFloat("###speed_control", &move_speed,
					0.001f, 10.0f, "%.3f unitless")) {
					m_event_dispatcher_handle->emitSignal(Event("movement_speed_changed"), move_speed);
				};

				ImGui::TableNextRow();
				ImGui::TableSetColumnIndex(0);
				ImGui::Text("Vertical FOV");
				ImGui::TableSetColumnIndex(1);
				ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x);
				if (ImGui::SliderAngle("###fov_control", &fov_y_radians, 0.0, 120.0)) {
					m_event_dispatcher_handle->emitSignal(Event("fov_changed"), fov_y_radians);
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
				m_event_dispatcher_handle->emitSignal(Event("exposure_changed"), camera_values);
			};
		};

		//integrator edit
		ImGui::Separator();
		if (ImGui::CollapsingHeader("Integrator Edit")) {
			KittlesPT::RendererSettings pt_settings = m_shared_data_handle->renderer_settings;
			bool pt_settings_updated = false;

			if (ImGui::BeginTable("integratoredittable", 2)) {
				ImGui::TableSetupColumn("A0", 0, 0.4f);
				ImGui::TableSetupColumn("A1", 0);

				ImGui::TableNextRow();
				ImGui::TableSetColumnIndex(0);
				ImGui::Text("Max Path Depth");
				ImGui::TableSetColumnIndex(1);
				ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x);
				pt_settings_updated |= ImGui::SliderInt("###max_bounces", &pt_settings.integrator_max_ray_depth, 0, 32);
				ImGui::EndTable();

				pt_settings_updated |= ImGui::Checkbox("Enable AutoExposure", &pt_settings.tonemapper_enable_auto_exposure);
				pt_settings_updated |= ImGui::Checkbox("Enable TAA", &pt_settings.integrator_use_temporal_accumulation);
				pt_settings_updated |= ImGui::Checkbox("Enable Post Effects", &pt_settings.postprocess_enable_effects);
				pt_settings_updated |= ImGui::Checkbox("Enable Upscaling", &pt_settings.upscale_enable);

				pt_settings_updated |= ImGui::Checkbox("Generate Veiling Luminance(Bloom)", &pt_settings.bloom_generate_bloom);
				ImGui::Indent();
				pt_settings_updated |= ImGui::Checkbox("Use Karis Average", &pt_settings.bloom_use_karis_average);
				if (ImGui::BeginTable("bloomedittable", 2)) {
					ImGui::TableSetupColumn("A0", 0, 0.8f);
					ImGui::TableSetupColumn("A1", 0);

					ImGui::TableNextRow();
					ImGui::TableSetColumnIndex(0);
					ImGui::Text("Bloom Blend Factor");
					ImGui::TableSetColumnIndex(1);
					ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x);
					pt_settings_updated |= ImGui::SliderFloat("###bloomblendfac", &pt_settings.bloom_final_blend, 0.0f, 1.0f);

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
				m_event_dispatcher_handle->emitSignal(Event("pathtracer_settings_changed"), pt_settings);
			}
		}

		//environment edit
		ImGui::Separator();
		if (ImGui::CollapsingHeader("Environment Edit"))
		{
			KittlesPT::ProceduralEnvironmentSettings env_settings = m_shared_data_handle->environment_settings;
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
				env_updated |= ImGui::SliderFloat("###sun_radiance", &env_settings.sun_emission_factor, 0, 6e5f,
					"%.3f nits", ImGuiSliderFlags_Logarithmic);

				ImGui::TableNextRow();
				ImGui::TableSetColumnIndex(0);
				ImGui::Text("Sun Angular Diameter");
				ImGui::TableSetColumnIndex(1);
				ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x);
				env_updated |= ImGui::SliderAngle("###sun_angular_diameter", &env_settings.sun_angular_diameter_rad, 0, 45,
					"%.1f deg", ImGuiSliderFlags_Logarithmic);

				ImGui::TableNextRow();
				ImGui::TableSetColumnIndex(0);
				ImGui::Text("Sun Altitude Theta");
				ImGui::TableSetColumnIndex(1);
				ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x);
				env_updated |= ImGui::SliderAngle("###sun_altitude_theta", &env_settings.sun_theta_rad, -20, 90);

				ImGui::TableNextRow();
				ImGui::TableSetColumnIndex(0);
				ImGui::Text("Sun Position Phi");
				ImGui::TableSetColumnIndex(1);
				ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x);
				env_updated |= ImGui::SliderAngle("###sun_phi", &env_settings.sun_phi_rad, 0, 360);

				ImGui::EndTable();
			}

			if (env_updated) {
				m_event_dispatcher_handle->emitSignal(Event("environment_settings_changed"), env_settings);
			}
		}

		//geometry edit
		ImGui::Separator();
		if (ImGui::CollapsingHeader("Geometry Edit")) {
			bool transform_updated = false;
			MeshObject editable_mesh;
			if (ImGui::BeginTable("geometryedittable", 2))
			{
				ImGui::TableSetupColumn("A0", 0, 0.6f);
				ImGui::TableSetupColumn("A1", 0);

				ImGui::TableNextRow();
				ImGui::TableSetColumnIndex(0);
				ImGui::Text("Mesh Instance ID:");
				ImGui::TableSetColumnIndex(1);
				ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x);
				if (ImGui::SliderInt("###mesh_selection", &m_shared_data_handle->editable_mesh_idx,
					0, static_cast<int32_t>(m_shared_data_handle->meshes_count - 1))) {
					m_event_dispatcher_handle->emitSignal(Event("mesh_changed"), true);
				};
				editable_mesh = m_shared_data_handle->editable_mesh_object;

				{
					ImGui::TableNextRow();
					ImGui::TableSetColumnIndex(0);
					ImGui::Text("Translation");
					ImGui::TableSetColumnIndex(1);
					ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x);
					transform_updated |= ImGui::DragFloat3("###translate", glm::value_ptr(editable_mesh.translation), 0.05f);
				}
				{
					static bool uniform_scale = true;
					ImGui::TableNextRow();
					ImGui::TableSetColumnIndex(0);
					ImGui::Text("Use uniform scale");
					ImGui::TableSetColumnIndex(1);
					ImGui::Checkbox("###uniformscale", &uniform_scale);

					ImGui::TableNextRow();
					ImGui::TableSetColumnIndex(0);
					ImGui::Text("Scale");
					ImGui::TableSetColumnIndex(1);
					ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x);
					if (uniform_scale) {
						if (ImGui::DragFloat("###scale", &editable_mesh.scale.x, 0.05f)) {
							transform_updated |= true;
							editable_mesh.scale.z = editable_mesh.scale.y = editable_mesh.scale.x;
						};
					}
					else {
						transform_updated |= ImGui::DragFloat3("###scale", glm::value_ptr(editable_mesh.scale), 0.05f);
					}
				}
				{
					ImGui::TableNextRow();
					ImGui::TableSetColumnIndex(0);
					ImGui::Text("Rotation");
					ImGui::TableSetColumnIndex(1);
					ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x);
					transform_updated |= ImGui::DragFloat3("###rotation", glm::value_ptr(editable_mesh.rotation), 1.0f);
				}

				ImGui::EndTable();
			}
			if (transform_updated)
			{
				//printf("%.3f | %.3f | %.3f\n", editable_mesh.translation.x, editable_mesh.translation.y, editable_mesh.translation.z);
				m_event_dispatcher_handle->emitSignal(Event("mesh_updated"), editable_mesh);
			}
		}

		//material edit
		ImGui::Separator();
		if (ImGui::CollapsingHeader("Material Edit"))
		{
			bool material_updated = false;
			KittlesPT::MaterialSceneEntity material;

			if (ImGui::BeginTable("materialedittable", 2))
			{
				ImGui::TableSetupColumn("A0", 0, 0.6f);
				ImGui::TableSetupColumn("A1", 0);

				ImGui::TableNextRow();
				ImGui::TableSetColumnIndex(0);
				ImGui::Text("Material ID:");
				ImGui::TableSetColumnIndex(1);
				ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x);
				if (ImGui::SliderInt("###material_selection", &m_shared_data_handle->editable_material_idx,
					0, int32_t(m_shared_data_handle->materials_count - 1)))
				{
					m_event_dispatcher_handle->emitSignal(Event("material_changed"), true);
				};

				material = m_shared_data_handle->editable_material;

				ImGui::TableNextRow();
				ImGui::TableSetColumnIndex(0);
				ImGui::Text("Albedo Factor");
				ImGui::TableSetColumnIndex(1);
				ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x);
				material_updated |= ImGui::ColorEdit3("###albedo", &material.albedo_factor.r);

				ImGui::TableNextRow();
				ImGui::TableSetColumnIndex(0);
				ImGui::Text("Metallness Factor");
				ImGui::TableSetColumnIndex(1);
				ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x);
				material_updated |= ImGui::SliderFloat("###metallicity", &material.metallic_factor, 0.0f, 1.0f);

				ImGui::TableNextRow();
				ImGui::TableSetColumnIndex(0);
				ImGui::Text("Roughness Factor");
				ImGui::TableSetColumnIndex(1);
				ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x);
				material_updated |= ImGui::SliderFloat("###roughness", &material.roughness_factor, 0.0f, 1.0f);

				ImGui::TableNextRow();
				ImGui::TableSetColumnIndex(0);
				ImGui::Text("Transmission Factor");
				ImGui::TableSetColumnIndex(1);
				ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x);
				material_updated |= ImGui::SliderFloat("###transmission", &material.transmission_factor, 0, 1);

				ImGui::TableNextRow();
				ImGui::TableSetColumnIndex(0);
				ImGui::Text("IOR");
				ImGui::TableSetColumnIndex(1);
				ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x);
				material_updated |= ImGui::SliderFloat("###ior", &material.ior, 1, 3);

				ImGui::TableNextRow();
				ImGui::TableSetColumnIndex(0);
				ImGui::Text("Emission color factor");
				ImGui::TableSetColumnIndex(1);
				ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x);
				material_updated |= ImGui::ColorEdit3("###emcol", &material.emission_factor.r);

				ImGui::TableNextRow();
				ImGui::TableSetColumnIndex(0);
				ImGui::Text("Emission scalar");
				ImGui::TableSetColumnIndex(1);
				ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x);
				material_updated |= ImGui::SliderFloat("###emscale", &material.emission_scale_nits, 0.0f, 1.0e6f,
					"%.3f nits", ImGuiSliderFlags_Logarithmic);

				ImGui::TableNextRow();
				ImGui::TableSetColumnIndex(0);
				ImGui::Text("Normal map scalar");
				ImGui::TableSetColumnIndex(1);
				ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x);
				material_updated |= ImGui::SliderFloat("###nrmscale", &material.normal_scale, 0.0f, 1.0f);

				ImGui::EndTable();
			}

			if (material_updated)
			{
				m_event_dispatcher_handle->emitSignal(Event("material_updated"), material);
			}
		}
	}
}