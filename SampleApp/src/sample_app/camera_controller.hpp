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

	private:
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