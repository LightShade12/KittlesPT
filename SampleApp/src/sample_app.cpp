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
	ImGui::ShowDemoWindow();

	developer_window.draw(m_window_ctx_handle, "Developer Menu");
}

void SampleAppWindow::updateUI()
{
	developer_window.updateUI();
}