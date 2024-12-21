#include "camera_controller.hpp"
#include "GLFW/glfw3.h"
#include "glm/gtc/matrix_transform.hpp"
#include "glm/gtc/quaternion.hpp"
#define GLM_ENABLE_EXPERIMENTAL
#include "glm/gtx/quaternion.hpp"

namespace SampleApp
{
	constexpr static glm::vec3 global_up(0, 1, 0);
	static glm::vec2 last_mouse_pos;

	bool CameraController::processInput(GLFWwindow* window_ctx, float delta_ts_ms)
	{
		double xpos, ypos;
		glfwGetCursorPos(window_ctx, &xpos, &ypos);
		glm::vec2 mouse_pos = { xpos,ypos };
		glm::vec2 mouse_delta = (mouse_pos - last_mouse_pos) * m_mouse_sensitivity;
		last_mouse_pos = mouse_pos;

		//if not held down; no movement possible; return;
		if (glfwGetMouseButton(window_ctx, GLFW_MOUSE_BUTTON_RIGHT) != GLFW_PRESS)
		{
			glfwSetInputMode(window_ctx, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
			m_moved = false;
			return m_moved;
		}

		glfwSetInputMode(window_ctx, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

		m_moved = false;

		float delta_ts = delta_ts_ms;

		//=======================================================================================
		//KEYBOARD INPUT PROCESSING
		//=======================================================================================

		if (glfwGetKey(window_ctx, GLFW_KEY_W) == GLFW_PRESS)//FORWARD
		{
			position += m_movement_speed * delta_ts * forward; m_moved |= true;
		}
		if (glfwGetKey(window_ctx, GLFW_KEY_S) == GLFW_PRESS)//BACK
		{
			position -= m_movement_speed * delta_ts * forward; m_moved |= true;
		}
		if (glfwGetKey(window_ctx, GLFW_KEY_A) == GLFW_PRESS)//LEFT
		{
			position -= m_movement_speed * delta_ts * right; m_moved |= true;
		}
		if (glfwGetKey(window_ctx, GLFW_KEY_D) == GLFW_PRESS)//RIGHT
		{
			position += m_movement_speed * delta_ts * right; m_moved |= true;
		}
		if (glfwGetKey(window_ctx, GLFW_KEY_E) == GLFW_PRESS)//UP
		{
			position += m_movement_speed * delta_ts * global_up; m_moved |= true;
		}
		if (glfwGetKey(window_ctx, GLFW_KEY_Q) == GLFW_PRESS)//DOWN
		{
			position -= m_movement_speed * delta_ts * global_up; m_moved |= true;
		}

		//=======================================================================================
		//MOUSE INPUT PROCESSING
		//=======================================================================================

		if (mouse_delta.x != 0.0f || mouse_delta.y != 0.0f)
		{
			float pitch_delta = mouse_delta.y * m_rotation_speed;
			float yaw_delta = mouse_delta.x * m_rotation_speed;

			glm::quat q = glm::normalize(glm::cross(glm::angleAxis(-pitch_delta, right),
				glm::angleAxis(-yaw_delta, global_up)));

			forward = glm::normalize(glm::rotate(q, forward));
			right = normalize(glm::cross(forward, global_up));
			up = normalize(glm::cross(right, forward));

			m_moved |= true;
		}

		return m_moved;
	}

	glm::mat4 CameraController::getViewMatrix()
	{
		glm::mat4 view
		(
			glm::vec4(right, 0),
			glm::vec4(up, 0),
			glm::vec4(forward, 0),
			glm::vec4(position, 1)
		);

		return view;
	}

	float CameraController::getVerticalFOV_Radians()
	{
		return m_fov_y_rad;
	}
}