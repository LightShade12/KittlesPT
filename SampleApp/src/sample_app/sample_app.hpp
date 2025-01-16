#pragma once
#include "gui_application.hpp"
#include "camera_controller.hpp"
#include "widgets.hpp"
#include "event_dispatcher.hpp"
#include "shared_state.hpp"

#include "kittlesPT/kittlesPT.hpp"

namespace SampleApp
{
	class SampleAppWindow : public SampleAppGUI::GUIWindow
	{
	public:

		using GUIWindow::GUIWindow;

		void onCreate() override;
		void onDestroy() override;

		void renderUI() override;

		void updateUI() override;

	public:
		ApplicationData m_application_data;
		EventDispatcher m_event_dispatcher;
		GLTexture m_viewport_texture;
		DeveloperWindow m_developer_window;
		BackgroundTexture m_viewport;
		CameraController m_camera;
		KittlesPT::Renderer m_renderer;
	};
}