#include "sample_app.hpp"

#include "glm/glm.hpp"
#include "glm/gtc/matrix_transform.hpp"
#include "glm/gtc/quaternion.hpp"
#define GLM_ENABLE_EXPERIMENTAL
#include "glm/gtx/quaternion.hpp"

void DeveloperWindow::updateUI()
{
	std::chrono::time_point<std::chrono::steady_clock> current_frame_time_point = std::chrono::high_resolution_clock::now();
	delta_time_secs = current_frame_time_point - last_frame_time_point;

	last_frame_time_point = current_frame_time_point;
}

void DeveloperWindow::renderUI()
{
	ImGui::Text("Delta ms(last frame): %.3f ms", delta_time_secs.count() * 1000.0f);
	ImGui::Text("FPS(last frame): %.3f ms", 1000.0f / (delta_time_secs.count() * 1000.0f));
	average_fps = glm::mix(average_fps, 1000.0f / (delta_time_secs.count() * 1000.0f), 0.01f);
	ImGui::Text("EMA FPS: %.3f ms", average_fps);
	ImGui::Text("Runtime secs: %.3f s",
		std::chrono::duration_cast<std::chrono::duration<float>>(last_frame_time_point.time_since_epoch()).count() - start_time_secs);
}

void SampleAppWindow::onCreate()
{
	m_viewport_texture.init(m_window_width, m_window_height);
	m_renderer.init();
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

	//TODO: background gui class
	{
		int winposx, winposy;
		glfwGetWindowPos(m_window_ctx_handle, &winposx, &winposy);
		ImGui::SetNextWindowPos(ImVec2(winposx, winposy));
		ImGui::SetNextWindowSize(ImVec2(m_window_width, m_window_height));

		ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.20f, 0.20f, 0.20f, 1.0f));
		ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
		ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
		ImGui::Begin("###viewport", nullptr,
			ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize |
			ImGuiWindowFlags_NoDocking | ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);
		if (m_viewport_texture.isValid())
		{
			ImGui::Image((void*)m_viewport_texture.m_GL_texture_name,
				ImVec2(m_window_width, m_window_height), { 0,1 }, { 1,0 });
		}

		ImGui::End();
		ImGui::PopStyleVar(2);
		ImGui::PopStyleColor(1);
	}

	//ImGui::ShowDemoWindow();
	developer_window.draw(m_window_ctx_handle, "Developer Menu");
}

void SampleAppWindow::updateUI()
{
	developer_window.updateUI();

	constexpr glm::vec3 global_up(0, 1, 0);
	static glm::vec2 last_mouse_pos;

	double xpos, ypos;
	glfwGetCursorPos(m_window_ctx_handle, &xpos, &ypos);

	glm::vec2 mouse_pos = { xpos,ypos };
	glm::vec2 mouse_delta = (mouse_pos - last_mouse_pos) * 0.002f;//TODO: add storage for literal here
	last_mouse_pos = mouse_pos;

	if (glfwGetMouseButton(m_window_ctx_handle, GLFW_MOUSE_BUTTON_RIGHT) != GLFW_PRESS)
	{
		glfwSetInputMode(m_window_ctx_handle, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
		m_camera.moved = false;
		return;
	}
	glfwSetInputMode(m_window_ctx_handle, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
	bool moved = false;

	float delta_ts = developer_window.getDeltaTS() / 1.0f;
	//delta_ts = 0.3f;

	if (glfwGetKey(m_window_ctx_handle, GLFW_KEY_W) == GLFW_PRESS)//FORWARD
	{
		m_camera.position += m_camera.movement_speed * delta_ts * m_camera.forward; moved |= true;
	}
	if (glfwGetKey(m_window_ctx_handle, GLFW_KEY_S) == GLFW_PRESS)//BACK
	{
		m_camera.position -= m_camera.movement_speed * delta_ts * m_camera.forward; moved |= true;
	}
	if (glfwGetKey(m_window_ctx_handle, GLFW_KEY_A) == GLFW_PRESS)//LEFT
	{
		m_camera.position -= m_camera.movement_speed * delta_ts * m_camera.right; moved |= true;
	}
	if (glfwGetKey(m_window_ctx_handle, GLFW_KEY_D) == GLFW_PRESS)//RIGHT
	{
		m_camera.position += m_camera.movement_speed * delta_ts * m_camera.right; moved |= true;
	}
	if (glfwGetKey(m_window_ctx_handle, GLFW_KEY_E) == GLFW_PRESS)//UP
	{
		m_camera.position += m_camera.movement_speed * delta_ts * global_up; moved |= true;
	}
	if (glfwGetKey(m_window_ctx_handle, GLFW_KEY_Q) == GLFW_PRESS)//DOWN
	{
		m_camera.position -= m_camera.movement_speed * delta_ts * global_up; moved |= true;
	}

	if (mouse_delta.x != 0.0f || mouse_delta.y != 0.0f)
	{
		float pitch_delta = mouse_delta.y * m_camera.rotation_speed;
		float yaw_delta = mouse_delta.x * m_camera.rotation_speed;

		glm::quat q = glm::normalize(glm::cross(glm::angleAxis(-pitch_delta, m_camera.right),
			glm::angleAxis(-yaw_delta, global_up)));

		m_camera.forward = glm::normalize(glm::rotate(q, m_camera.forward));
		m_camera.right = normalize(glm::cross(m_camera.forward, global_up));
		m_camera.up = normalize(glm::cross(m_camera.right, m_camera.forward));

		moved = true;
	}

	m_camera.moved = moved;

	if (m_camera.moved)
	{
		glm::mat4 view
		(
			glm::vec4(m_camera.right, 0),
			glm::vec4(m_camera.up, 0),
			glm::vec4(m_camera.forward, 0),
			glm::vec4(m_camera.position, 1)
		);
		glm::mat4 proj = glm::perspectiveFovLH(m_camera.fov_y_rad, float(m_window_width), float(m_window_height), 1.f, 100.f);
		m_renderer.setView(proj, glm::inverse(view));
	}
}