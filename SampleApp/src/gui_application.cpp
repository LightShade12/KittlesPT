#include "gui_application.hpp"
#include "sample_app.hpp"//TOOO: backend does not need to know of implementations

void ToggleableSideWindow::draw(GLFWwindow* glfw_main_window, const char* window_title)
{
	glfwGetFramebufferSize(glfw_main_window, &m_glfw_window_width, &m_glfw_window_height);
	glfwGetWindowPos(glfw_main_window, &m_glfw_window_pos_x, &m_glfw_window_pos_y);

	ImGui::SetNextWindowPos({ m_glfw_window_pos_x + m_glfw_window_width - ((!m_is_toggled) ? m_window_size.x : m_collapsed_window_size.x),
							  m_glfw_window_pos_y + (m_glfw_window_height / 2.0f) - (((!m_is_toggled) ? m_window_size.y : m_collapsed_window_size.y) / 2.0f) });

	if (!m_is_toggled)
	{
		ImGui::SetNextWindowSize(ImVec2(
			glm::clamp(m_window_size.x, 0.0f, (float)m_glfw_window_width),
			glm::clamp(m_window_size.y, 0.0f, (float)m_glfw_window_height)));
		ImGui::Begin(window_title);
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

GUIWindow::GUIWindow(const char* title, int initial_width, int initial_height, const char* glsl_version_formatted)
	:width(initial_width), height(initial_height)
{
	m_window_ctx_handle = glfwCreateWindow(width, height, title, NULL, NULL);
	if (!isValid())
	{
		glfwTerminate();
		exit(EXIT_FAILURE);
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

	ImGuiThemes::DarkRudra();

	ImGui_ImplOpenGL3_Init(glsl_version_formatted);
	ImGui_ImplGlfw_InitForOpenGL(m_window_ctx_handle, true);
}

void GUIWindow::processAndDraw()
{
	if (glfwGetWindowAttrib(m_window_ctx_handle, GLFW_ICONIFIED) != 0)
	{
		ImGui_ImplGlfw_Sleep(10);
		return;
	}

	glClear(GL_COLOR_BUFFER_BIT);

	glfwGetFramebufferSize(m_window_ctx_handle, &width, &height);
	glViewport(0, 0, width, height);

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
};

static void glfw_error_callback(int error, const char* description)
{
	fprintf(stderr, "GLFW Error: %s\n", description);
}

void GUIApplication::init()
{
	glfwSetErrorCallback(glfw_error_callback);

	if (!glfwInit()) {
		exit(EXIT_FAILURE);
	}

	//MAJOR=4 MINOR=6 on my system
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	glfwWindowHint(GLFW_MAXIMIZED, GLFW_TRUE);

	main_window = std::make_shared<SampleAppWindow>("main_window",
		window_settings.initial_window_width,
		window_settings.initial_window_height,
		window_settings.glsl_version_formatted);
};

void GUIApplication::run()
{
	while (!glfwWindowShouldClose(main_window->m_window_ctx_handle))
	{
		main_window->processAndDraw();
	}
};

void GUIApplication::destroy()
{
	ImGui_ImplOpenGL3_Shutdown();
	ImGui_ImplGlfw_Shutdown();
	ImGui::DestroyContext();

	glfwDestroyWindow(main_window->m_window_ctx_handle);
	glfwTerminate();
};