#pragma once
#include "gui_application.hpp"
#include "gui_widgets.hpp"
#include "texture.hpp"

#include <chrono>

namespace SampleApp
{
	class DeveloperWindow : public SampleAppGUI::ToggleableSideWindow
	{
	public:
		void updateUI() override;
		float getDeltaTS() const { return delta_time_secs.count(); };
	private:

		void renderUI() override;

	private:
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