#pragma once
#include "gui_application.hpp"
#include "camera_controller.hpp"
#include "mesh_object.hpp"
#include "widgets.hpp"
#include "event_dispatcher.hpp"
#include "shared_state.hpp"

#include "kittlesPT/kittlesPT.hpp"

#include <vector>
#include <unordered_map>

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

	private:
		void loadSceneFile(const char* file_path);
		void registerListeners();
	public:
		std::vector<MeshObject> m_meshes;
		std::unordered_map <std::string, GLTexture> m_textures;

		ApplicationData m_application_data;
		EventDispatcher m_event_dispatcher;
		GLTexture m_viewport_texture;
		DeveloperWindow m_developer_window;
		BackgroundTexture m_viewport;
		CameraController m_camera;
		KittlesPT::Renderer m_renderer;
	};
}