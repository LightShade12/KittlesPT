#pragma once
#include "imgui/imgui.h"
#include "imgui/backends/imgui_impl_glfw.h"
#include "imgui/backends/imgui_impl_opengl3.h"

struct GLFWwindow;

class ToggleableSideWindow
{
public:

	ToggleableSideWindow() :m_window_size(256, 512), m_collapsed_window_size(0, 0) {};

	void draw(GLFWwindow* glfw_main_window, const char* window_title);

	//custom window content updating; manually called
	virtual void updateUI() = 0;

protected:
	//custom window content rendering; called by draw()
	virtual void renderUI() = 0;

private:
	int m_glfw_window_pos_x = 0, m_glfw_window_pos_y = 0;
	int m_glfw_window_width = 0, m_glfw_window_height = 0;
	ImVec2 m_collapsed_window_size;
	bool m_is_toggled = false;
	ImVec2 m_window_size;
};