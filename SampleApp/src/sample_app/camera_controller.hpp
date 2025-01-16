#pragma once
#include "glm/glm.hpp"

struct GLFWwindow;

namespace SampleApp
{
	class CameraController
	{
	public:
		CameraController() = default;

		bool processInput(GLFWwindow* window_ctx, float delta_ts_ms);
		glm::mat4 getViewMatrix();
		float getVerticalFOV_Radians();
		float getISO();
		float getShutterSpeedSec() { return m_shutter_sec; };
		float getApertureNumber() { return m_aperture_num; };

		void setISO(float ISO);
		void setShutterSpeedSec(float secs);
		void setApertureNumber(float f_num);
		void setVerticalFOV_Radians(float fov_y_rad);

	private:
		float m_ISO = 100.0f;//NOTE: does not belong here cleanly; this class should only contain transform relative data
		float m_shutter_sec = 1.0f / 20.0f;
		float m_aperture_num = 11.0f;
		float m_fov_y_rad = glm::radians(90.0f);
		float m_movement_speed = 5.0f;
		float m_rotation_speed = 0.8f;
		float m_mouse_sensitivity = 0.002f;
		bool m_moved = false;

		glm::vec3 position = glm::vec3(0.0f);
		glm::vec3 forward = { 0,0,-1 };
		glm::vec3 up = { 0,1,0 };
		glm::vec3 right = { 1,0,0 };
	};
}