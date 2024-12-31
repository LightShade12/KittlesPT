#include "gui_application.hpp"
#include "imgui/imgui.h"
#include "imgui/backends/imgui_impl_glfw.h"
#include "imgui/backends/imgui_impl_opengl3.h"

#include "glad/include/glad/glad.h"
#define GLFW_INCLUDE_NONE //glad loader instead of local gl
#include "glfw/include/GLFW/glfw3.h"

#include "glm/glm.hpp"
#include "imgui_themes.hpp"

namespace SampleAppGUI
{
	AppStatus GUIWindow::init(WindowConfig window_config)
	{
		m_window_width = window_config.initial_window_width;
		m_window_height = window_config.initial_window_height;
		m_window_ctx_handle = glfwCreateWindow(m_window_width, m_window_height, window_config.window_title.c_str(),
			NULL, NULL);

		if (!isValid())
		{
			return AppStatus::FAILURE;
		}

		setCurrent();
		gladLoadGLLoader((GLADloadproc)glfwGetProcAddress);
		glfwSwapInterval(1);

		IMGUI_CHECKVERSION();
		m_imgui_ctx_handle = ImGui::CreateContext();
		ImGui::SetCurrentContext(m_imgui_ctx_handle);
		ImGuiIO& io = ImGui::GetIO();
		io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
		io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
		io.ConfigFlags |= ImGuiConfigFlags_ViewportsEnable;

		ImGui_ImplOpenGL3_Init(window_config.glsl_version_formatted);
		ImGui_ImplGlfw_InitForOpenGL(m_window_ctx_handle, true);

		glClearColor(0.f, 0.24f, 0.3f, 1.f);
		onCreate();

		return AppStatus::SUCCESS;
	}

	void GUIWindow::processAndDraw()
	{
		if (glfwGetWindowAttrib(m_window_ctx_handle, GLFW_ICONIFIED) != 0)
		{
			ImGui_ImplGlfw_Sleep(10);
			return;
		}

		glClear(GL_COLOR_BUFFER_BIT);

		glfwGetFramebufferSize(m_window_ctx_handle, &m_window_width, &m_window_height);
		glViewport(0, 0, m_window_width, m_window_height);

		ImGui_ImplOpenGL3_NewFrame();
		ImGui_ImplGlfw_NewFrame();
		ImGui::NewFrame();

		renderUI();

		ImGui::Render();
		ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

		if (ImGui::GetIO().ConfigFlags & ImGuiConfigFlags_ViewportsEnable)
		{
			GLFWwindow* backup_current_context = glfwGetCurrentContext();
			ImGui::UpdatePlatformWindows();
			ImGui::RenderPlatformWindowsDefault();
			glfwMakeContextCurrent(backup_current_context);
		}

		updateUI();
		glfwPollEvents();

		glfwSwapBuffers(m_window_ctx_handle);
	}

	void GUIWindow::setCurrent()
	{
		glfwMakeContextCurrent(m_window_ctx_handle);
		if (m_imgui_ctx_handle != nullptr)
		{
			ImGui::SetCurrentContext(m_imgui_ctx_handle);
		}
	}
	bool GUIWindow::shouldClose()
	{
		return glfwWindowShouldClose(m_window_ctx_handle);
	}

	void GUIWindow::destroy()
	{
		//order matters
		onDestroy();
		ImGui_ImplOpenGL3_Shutdown();
		ImGui_ImplGlfw_Shutdown();
		ImGui::DestroyContext();
		glfwDestroyWindow(m_window_ctx_handle);
	};

	static void glfw_error_callback(int error, const char* description)
	{
		fprintf(stderr, "GLFW Error: %s\n", description);
	}

	void GUIApplication::init(std::shared_ptr<GUIWindow> window)
	{
		main_window = window;

		glfwSetErrorCallback(glfw_error_callback);

		if (!glfwInit()) {
			exit(EXIT_FAILURE);
		}

		//MAJOR=4 MINOR=6 on my system
		glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
		glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
		glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
		//glfwWindowHint(GLFW_MAXIMIZED, GLFW_TRUE);

		AppStatus stat = main_window->init(window_settings);
		if (stat == AppStatus::FAILURE)
		{
			printf("failed to create window\n");
			glfwTerminate();
			exit(EXIT_FAILURE);
		}
	};

	void GUIApplication::run()
	{
		while (!main_window->shouldClose())
		{
			main_window->processAndDraw();
		}
	};

	void GUIApplication::destroy()
	{
		main_window->destroy();
		glfwTerminate();
	};
}