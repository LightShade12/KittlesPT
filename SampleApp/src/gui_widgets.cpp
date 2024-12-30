#include "gui_widgets.hpp"
#include "glad/include/glad/glad.h"
#define GLFW_INCLUDE_NONE //glad loader instead of local gl
#include "glfw/include/GLFW/glfw3.h"

#include <algorithm>

namespace SampleAppGUI
{
	void ToggleableSideWindow::draw(GLFWwindow* glfw_main_window, const char* window_title)
	{
		glfwGetFramebufferSize(glfw_main_window, &m_glfw_window_width, &m_glfw_window_height);
		glfwGetWindowPos(glfw_main_window, &m_glfw_window_pos_x, &m_glfw_window_pos_y);

		ImGui::SetNextWindowPos({ m_glfw_window_pos_x + m_glfw_window_width - ((!m_is_toggled) ? m_window_size.x : m_collapsed_window_size.x),
								  m_glfw_window_pos_y + (m_glfw_window_height / 2.0f) - (((!m_is_toggled) ? m_window_size.y : m_collapsed_window_size.y) / 2.0f) });

		if (!m_is_toggled)
		{
			ImGui::SetNextWindowSize(ImVec2(
				std::clamp(m_window_size.x, 0.0f, (float)m_glfw_window_width),
				std::clamp(m_window_size.y, 0.0f, (float)m_glfw_window_height)));
			ImGui::Begin(window_title, nullptr, ImGuiWindowFlags_NoScrollbar);
			if (ImGui::Button("Hide window")) { m_is_toggled = !m_is_toggled; }
			ImGui::Separator();

			renderUI();

			m_window_size = ImGui::GetWindowSize();
			ImGui::End();
		}
		else
		{
			ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 10.0);
			ImGui::PushStyleVar(ImGuiStyleVar_Alpha, 0.75);
			ImGui::Begin("###hidden_toggle_window", (bool*)0,
				ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize);
			m_collapsed_window_size = ImGui::GetWindowSize();
			if (ImGui::Button("Show window")) { m_is_toggled = !m_is_toggled; }
			ImGui::End();
			ImGui::PopStyleVar(2);
		}
	}
}