#include "widgets.hpp"

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
	ImGui::Text("FPS(last frame): %.3f ms", 1000.0f / (delta_time_secs.count() * 1000.0f));
	average_fps = glm::mix(average_fps, 1000.0f / (delta_time_secs.count() * 1000.0f), 0.01f);
	ImGui::Text("EMA FPS: %.3f ms", average_fps);
	ImGui::Text("Runtime secs: %.3f s",
		std::chrono::duration_cast<std::chrono::duration<float>>(last_frame_time_point.time_since_epoch()).count() - start_time_secs);
}