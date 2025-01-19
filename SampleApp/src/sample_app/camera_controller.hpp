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
		glm::mat4 getViewMatrix() const;

		float getVerticalFOV_Radians() const { return m_fov_y_rad; };
		void setVerticalFOV_Radians(float fov_y_rad) { m_fov_y_rad = fov_y_rad; };

		float getAperture() const { return m_aperture_f_num; }
		void setAperture(float f_num) { m_aperture_f_num = f_num; }
		float getExposureCompensation() const { return m_exposure_compensation; }
		void setExposureCompensation(float exposure) { m_exposure_compensation = exposure; }
		int getISO() const { return m_ISO; }
		void setISO(int iso) { m_ISO = iso; }
		float getShutter() const { return m_shutter_secs; }
		void setShutter(float secs) { m_shutter_secs = secs; }

	private:
		float m_aperture_f_num = 2.8f;
		float m_shutter_secs = 0.01666f; //1/60th sec
		int m_ISO = 100;
		float m_exposure_compensation = 0.0f;

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