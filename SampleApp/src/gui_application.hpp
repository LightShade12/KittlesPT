#pragma once
#include "imgui/imgui.h"
#include "imgui/backends/imgui_impl_glfw.h"
#include "imgui/backends/imgui_impl_opengl3.h"

#include "glad/include/glad/glad.h"
#define GLFW_INCLUDE_NONE //glad loader instead of local gl
#include "glfw/include/GLFW/glfw3.h"

#include "glm/glm.hpp"
#include "imgui_themes.hpp"

#include <iostream>

class ToggleableSideWindow
{
public:

	ToggleableSideWindow() :m_window_size(256, 512), m_collapsed_window_size(0, 0) {};

	void draw(GLFWwindow* glfw_main_window, const char* window_title);

	//custom window content updating
	virtual void updateUI() = 0;

private:
	//custom window content rendering
	virtual void renderUI() = 0;

private:
	int m_glfw_window_pos_x = 0, m_glfw_window_pos_y = 0;
	int m_glfw_window_width = 0, m_glfw_window_height = 0;
	ImVec2 m_collapsed_window_size;
	bool m_is_toggled = false;
	ImVec2 m_window_size;
};

class GUIWindow {
public:

	GUIWindow(const char* title, int initial_width, int initial_height, const char* glsl_version_formatted);

	//custom gui content rendering
	virtual void renderUI() = 0;
	//custom gui content update handling
	virtual void updateUI() = 0;

	void processAndDraw();

	void setCurrent();

	bool isValid() { return m_window_ctx_handle != nullptr; }

public:
	int width = 0, height = 0;
	ImGuiContext* m_imgui_ctx_handle = nullptr;
	GLFWwindow* m_window_ctx_handle = nullptr;
};

class GUIApplication {
public:
	struct WindowConfig
	{
		const int initial_window_width = 640 + 16;
		const int initial_window_height = 700;
		const char* glsl_version_formatted = "#version 460";
	};

	void init();

	void run();

	void destroy();

public:

	WindowConfig window_settings;
	std::shared_ptr<GUIWindow> main_window;
};