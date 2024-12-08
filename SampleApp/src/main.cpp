#include "imgui/imgui.h"
#include "imgui/backends/imgui_impl_glfw.h"
#include "imgui/backends/imgui_impl_opengl3.h"

#include "glad/include/glad/glad.h"
#define GLFW_INCLUDE_NONE //glad loader instead of local gl
#include "glfw/include/GLFW/glfw3.h"

#include "glm/glm.hpp"

#include "imgui_themes.hpp"

#include <iostream>
#include <chrono>
#include <algorithm>
#include <functional>
#include <utility> // for std::forward

static void glfw_error_callback(int error, const char* description)
{
	fprintf(stderr, "GLFW Error: %s\n", description);
}

class ToggleableSideWindow
{
public:

	ToggleableSideWindow() :m_window_size(256, 512), m_collapsed_window_size(0, 0) {};

	void draw(GLFWwindow* glfw_main_window, const char* window_title, std::function<void()> lambda_function)
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
			ImGui::Begin(window_title);
			if (ImGui::Button("Hide window")) { m_is_toggled = !m_is_toggled; }
			ImGui::Separator();

			// Call the lambda function here
			if (lambda_function) {
				lambda_function();
			}

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

private:
	int m_glfw_window_pos_x = 0, m_glfw_window_pos_y = 0;
	int m_glfw_window_width = 0, m_glfw_window_height = 0;
	ImVec2 m_collapsed_window_size;
	bool m_is_toggled = false;
	ImVec2 m_window_size;
};

int main()
{
	fprintf(stdout, "Initializing app\n");

	glfwSetErrorCallback(glfw_error_callback);

	if (!glfwInit()) {
		exit(EXIT_FAILURE);
	}

	//MAJOR=4 MINOR=6 on my system
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	glfwWindowHint(GLFW_MAXIMIZED, GLFW_TRUE);
	const char* glsl_version = "#version 460";

	float start_time_secs = std::chrono::duration_cast<std::chrono::duration<float>>(
		std::chrono::high_resolution_clock::now().time_since_epoch()).count();
	std::chrono::time_point<std::chrono::steady_clock> last_frame_time_point;
	std::chrono::duration<float> delta_time_secs;
	float average_fps = 0;

	int glfw_window_width = 640 + 16;
	int glfw_window_height = 700;

	GLFWwindow* main_window = glfwCreateWindow(glfw_window_width, glfw_window_height, "MainWindow", NULL, NULL);
	
	if (!main_window) {
		glfwTerminate();
		exit(EXIT_FAILURE);
	}

	glfwMakeContextCurrent(main_window);
	gladLoadGLLoader((GLADloadproc)glfwGetProcAddress);
	glfwSwapInterval(1);

	int gl_ver_minor, gl_ver_major, gl_extensions_num;
	glGetIntegerv(GL_MAJOR_VERSION, &gl_ver_major);
	glGetIntegerv(GL_MINOR_VERSION, &gl_ver_minor);
	glGetIntegerv(GL_NUM_EXTENSIONS, &gl_extensions_num);
	printf("GL version: %d.%d\nVendor: %s\nRenderer: %s\n", gl_ver_major, gl_ver_minor, 
		glGetString(GL_VENDOR), glGetString(GL_RENDERER));
	printf("GLSL version: %s\n", glGetString(GL_SHADING_LANGUAGE_VERSION));
	printf("Extensions used: %d\n", gl_extensions_num);

	/*
	if (gl_extensions_num >= 0)
	{
		printf("Using extensions:\n");
		for (int i = 0; i < gl_extensions_num; i++) {
			printf("%s,", glGetStringi(GL_EXTENSIONS, i));
		}
	}
	*/

	// Setup Dear ImGui context
	IMGUI_CHECKVERSION();
	ImGui::CreateContext();
	ImGuiIO& io = ImGui::GetIO();
	io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
	io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
	io.ConfigFlags |= ImGuiConfigFlags_ViewportsEnable;

	//styling
	//ImGui::StyleColorsDark();
	ImGuiThemes::DarkRudra();

	// Setup Platform/Renderer backends
	ImGui_ImplOpenGL3_Init(glsl_version);
	ImGui_ImplGlfw_InitForOpenGL(main_window, true);
	//===========================================

	ToggleableSideWindow toggleable_developer_window;
	glClearColor(0.f, 0.24f, 0.3f, 1.f);

	fprintf(stdout, "starting mainloop..\n");
	
	while (!glfwWindowShouldClose(main_window))
	{
		glfwPollEvents();

		if (glfwGetWindowAttrib(main_window, GLFW_ICONIFIED) != 0)
		{
			ImGui_ImplGlfw_Sleep(10);
			continue;
		}

		glClear(GL_COLOR_BUFFER_BIT);

		glfwGetFramebufferSize(main_window, &glfw_window_width, &glfw_window_height);
		glViewport(0, 0, glfw_window_width, glfw_window_height);

		// Start the Dear ImGui frame
		ImGui_ImplOpenGL3_NewFrame();
		ImGui_ImplGlfw_NewFrame();
		ImGui::NewFrame();

		ImGui::ShowDemoWindow();
		//--------------------------------------

		toggleable_developer_window.draw(main_window, "Developer Menu",
			[&]()
			{
				ImGui::Text("Delta ms(last frame): %.3f ms", delta_time_secs.count() * 1000.0f);
				ImGui::Text("FPS(last frame): %.3f ms", 1000.0f / (delta_time_secs.count() * 1000.0f));
				average_fps = glm::mix(average_fps, 1000.0f / (delta_time_secs.count() * 1000.0f), 0.01f);
				ImGui::Text("EMA FPS: %.3f ms", average_fps);
				ImGui::Text("Runtime secs: %.3f s",
					std::chrono::duration_cast<std::chrono::duration<float>>(last_frame_time_point.time_since_epoch()).count() - start_time_secs);
			});
		//--------------------------------------

		ImGui::Render();
		ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

		if (ImGui::GetIO().ConfigFlags & ImGuiConfigFlags_ViewportsEnable)
		{
			GLFWwindow* backup_current_context = glfwGetCurrentContext();
			ImGui::UpdatePlatformWindows();
			ImGui::RenderPlatformWindowsDefault();
			glfwMakeContextCurrent(backup_current_context);
		}

		std::chrono::time_point<std::chrono::steady_clock> current_frame_time_point = std::chrono::high_resolution_clock::now();
		delta_time_secs = current_frame_time_point - last_frame_time_point;

		last_frame_time_point = current_frame_time_point;

		glfwSwapBuffers(main_window);
	}

	fprintf(stdout, "closing app\n");

	//===========================================
	ImGui_ImplOpenGL3_Shutdown();
	ImGui_ImplGlfw_Shutdown();
	ImGui::DestroyContext();

	glfwDestroyWindow(main_window);
	glfwTerminate();
}