#pragma once
#include "gui_application.hpp"

#include <chrono>

class DeveloperWindow : public ToggleableSideWindow
{
public:
	void updateUI() override;

private:

	void renderUI() override;

private:

	float start_time_secs = std::chrono::duration_cast<std::chrono::duration<float>>(
		std::chrono::high_resolution_clock::now().time_since_epoch()).count();
	std::chrono::time_point<std::chrono::steady_clock> last_frame_time_point;
	std::chrono::duration<float> delta_time_secs;
	float average_fps = 0;
};

class SampleAppWindow : public GUIWindow
{
public:

	using GUIWindow::GUIWindow;

	void renderUI() override;

	void updateUI() override;

public:

	DeveloperWindow developer_window;
};