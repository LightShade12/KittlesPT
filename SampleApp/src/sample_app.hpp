#pragma once
#include "gui_application.hpp"
#include "kittles_pt/kittles_pt.hpp"

#include <chrono>

class DeveloperWindow : public ToggleableSideWindow
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

struct GLTexture
{
	void init(int width, int height)
	{
		m_width = width; m_height = height;
		glGenTextures(1, &m_GL_texture_name);
		glBindTexture(GL_TEXTURE_2D, m_GL_texture_name);

		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);

		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, m_width, m_height, 0, GL_RGBA, GL_FLOAT, NULL);

		glBindTexture(GL_TEXTURE_2D, 0);
	}

	void resize(int width, int  height)
	{
		if (m_width == width && m_height == height)
		{
			return;
		}
		m_width = width; m_height = height;

		glBindTexture(GL_TEXTURE_2D, m_GL_texture_name);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, m_width, m_height, 0, GL_RGBA, GL_FLOAT, NULL);
		glBindTexture(GL_TEXTURE_2D, 0);
	}

	bool isValid() {
		return m_GL_texture_name != NULL;
	}

	void destroy() {
		glDeleteTextures(1, &m_GL_texture_name);
	}

	int m_width = 0, m_height = 0;
	GLuint m_GL_texture_name = NULL;
};

class Camera
{
public:
	Camera() = default;
	float fov_y_rad = glm::radians(90.0f);
	float movement_speed = 5.0f;
	float rotation_speed = 0.8f;

	bool moved = false;
	glm::vec3 position = glm::vec3(0.0f);
	glm::vec3 forward = { 0,0,-1 };
	glm::vec3 up = { 0,1,0 };
	glm::vec3 right = { 1,0,0 };
};

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
	Camera m_camera;
	KittlesPT::Renderer m_renderer;
};