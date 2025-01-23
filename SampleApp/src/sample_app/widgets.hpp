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
			event_dispatcher_ref = dispatcher;
			camera_controller_ref = camera;
			shared_data_ref = app_data;
		};
		float getDeltaTS_ms() const { return delta_time_secs.count(); };
	private:

		void renderUI() override;

	private:
		ApplicationData* shared_data_ref = nullptr;
		EventDispatcher* event_dispatcher_ref = nullptr;
		CameraController* camera_controller_ref = nullptr;
		float start_time_secs = std::chrono::duration_cast<std::chrono::duration<float>>(
			std::chrono::high_resolution_clock::now().time_since_epoch()).count();
		std::chrono::time_point<std::chrono::steady_clock> last_frame_time_point;
		std::chrono::duration<float> delta_time_secs;
		float average_fps = 0;
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