#include "sample_app.hpp"

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

void SampleAppWindow::renderUI()
{
	{
		int winposx, winposy;
		glfwGetWindowPos(m_window_ctx_handle, &winposx, &winposy);
		ImGui::SetNextWindowPos(ImVec2(winposx, winposy));
		ImGui::SetNextWindowSize(ImVec2(width, height));

		ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.20f, 0.20f, 0.20f, 1.0f));
		ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
		ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
		ImGui::Begin("###viewport", nullptr,
			ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize |
			ImGuiWindowFlags_NoDocking | ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);

		ImGui::Image(NULL, ImVec2(width, height));

		ImGui::End();
		ImGui::PopStyleVar(2);
		ImGui::PopStyleColor(1);
	}

	ImGui::ShowDemoWindow();
	developer_window.draw(m_window_ctx_handle, "Developer Menu");
}

void SampleAppWindow::updateUI()
{
	developer_window.updateUI();
}