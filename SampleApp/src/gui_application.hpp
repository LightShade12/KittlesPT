#pragma once
#include <iostream>

struct GLFWwindow;
struct ImGuiContext;

namespace SampleAppGUI
{
	enum class AppStatus
	{
		NONE = 0,
		FAILURE,
		SUCCESS
	};

	class GUIWindow
	{
	public:
		struct WindowConfig
		{
			const int initial_window_width = 640 + 16;
			const int initial_window_height = 700;
			const char* glsl_version_formatted = "#version 460";
			std::string window_title = "default title";
		};

		GUIWindow() = default;

	protected:
		virtual void onCreate() = 0;
		virtual void onDestroy() = 0;
		//custom gui content rendering
		virtual void renderUI() = 0;
		//custom gui content update handling
		virtual void updateUI() = 0;
	public:
		AppStatus init(WindowConfig window_config);//created because pure virtual function cannot run in constrcutor
		void processAndDraw();

		void setCurrent();

		bool shouldClose();
		void destroy();

		bool isValid() { return m_window_ctx_handle != nullptr; }
	public:
		int m_window_width = 0, m_window_height = 0;
		ImGuiContext* m_imgui_ctx_handle = nullptr;
		GLFWwindow* m_window_ctx_handle = nullptr;
	};

	class GUIApplication
	{
	public:

		void init(std::shared_ptr<GUIWindow> window);

		void run();

		void destroy();

	public:

		GUIWindow::WindowConfig window_settings;
		std::shared_ptr<GUIWindow> main_window;
	};
}