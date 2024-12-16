#pragma once
#include "gui_application.hpp"
#include "camera_controller.hpp"
#include "widgets.hpp"
#include "kittles_pt/kittles_pt.hpp"

class SampleAppWindow : public GUIWindow
{
public:

	using GUIWindow::GUIWindow;

	void onCreate() override;
	void onDestroy() override;

	void renderUI() override;

	void updateUI() override;

public:
	GLTexture m_viewport_texture;
	DeveloperWindow m_developer_window;
	BackgroundTexture m_viewport;
	Camera m_camera;
	KittlesPT::Renderer m_renderer;
};