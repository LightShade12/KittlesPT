#pragma once
#include "gui_application.hpp"
#include "gui_widgets.hpp"
#include "texture.hpp"
#include "event_dispatcher.hpp"
#include "camera_controller.hpp"
#include "shared_state.hpp"

#include <chrono>

namespace SampleApp
{
	class DeveloperWindow : public SampleAppGUI::ToggleableSideWindow
	{
	public:
		void updateUI() override;
		void init(EventDispatcher* dispatcher, CameraController* camera, ApplicationData* app_data)
		{
			m_event_dispatcher_handle = dispatcher;
			m_camera_handle = camera;
			m_shared_data_handle = app_data;
		};
		float getDeltaTS_ms() const { return m_delta_time_secs.count(); };
	private:

		void renderUI() override;

	private:
		ApplicationData* m_shared_data_handle = nullptr;
		EventDispatcher* m_event_dispatcher_handle = nullptr;
		CameraController* m_camera_handle = nullptr;
		float m_start_time_secs = std::chrono::duration_cast<std::chrono::duration<float>>(
			std::chrono::high_resolution_clock::now().time_since_epoch()).count();
		std::chrono::time_point<std::chrono::steady_clock> m_last_frame_time_point;
		std::chrono::duration<float> m_delta_time_secs;
		float m_average_fps = 0;
	};

	class BackgroundTexture
	{
	public:

		void init(GLTexture bg_texture);

		//no custom RenderUI() content
		void draw(GLFWwindow* window_ctx);

	private:
		GLTexture m_background_texture;
	};
}