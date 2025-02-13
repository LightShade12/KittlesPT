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

		void  setMovementSpeed(float speed) { m_movement_speed = speed; };
		float getMovementSpeed() const { return m_movement_speed; };

		void  setPosition(const glm::vec3& pos) { position = pos; };
		glm::vec3 getPosition() const { return position; };

		void  setLookAt(const glm::vec3& dir)
		{
			forward = dir;
			right = normalize(glm::cross(forward, global_up));
			up = normalize(glm::cross(right, forward));
		};

		float getVerticalFOV_Radians() const { return m_fov_y_rad; };
		void setVerticalFOV_Radians(float fov_y_rad) { m_fov_y_rad = fov_y_rad; };

		float getApertureF() const { return m_aperture_f_num; }
		void setAperture(float f_num) { m_aperture_f_num = f_num; }
		float getExposureCompensationEV() const { return m_exposure_compensation_ev; }
		void setExposureCompensationEV(float exposure_ev) { m_exposure_compensation_ev = exposure_ev; }
		int getISO() const { return m_ISO; }
		void setISO(int iso) { m_ISO = iso; }
		float getShutterSecs() const { return m_shutter_secs; }
		void setShutterSecs(float secs) { m_shutter_secs = secs; }

		float getBlackPointEV() const { return m_black_point_ev; }
		void setBlackPointEV(float black_point_ev) { m_black_point_ev = black_point_ev; }
		float getWhitePointEV() const { return m_white_point_ev; }
		void setWhitePointEV(float white_point_ev) { m_white_point_ev = white_point_ev; }

		static constexpr int ISO_MAX = 6400;
		static constexpr int ISO_MIN = 100;
		static constexpr float AP_F_MAX = 22.0f;//Aperture F-number max
		static constexpr float AP_F_MIN = 1.8f;
		static constexpr float SHUTTER_DENOM_MIN = 30.0f;
		static constexpr float SHUTTER_DENOM_MAX = 400.0f;

	private:
		constexpr static glm::vec3 global_up = glm::vec3(0, 1, 0);
		//default: DayTime preset
		float m_aperture_f_num = 11.0f;
		float m_exposure_compensation_ev = 0.0f;
		int m_ISO = 100;
		float m_shutter_secs = 0.004f; //1/60th sec
		float m_white_point_ev = 6.5f;
		float m_black_point_ev = -10.0f;

		float m_fov_y_rad = glm::radians(90.0f);

		float m_movement_speed = 5.0f;
		float m_rotation_speed = 0.8f;
		float m_mouse_sensitivity = 0.002f;
		bool m_moved = false;

		glm::vec3 position = glm::vec3(0.0f);
		glm::vec3 forward = { 0,0,-1 };//-Z forward
		glm::vec3 up = { 0,1,0 };
		glm::vec3 right = { 1,0,0 };
	};

	//DayTime preset
	inline void applySunnyDayCameraPreset(CameraController* cam)
	{
		/*
		A fast shutter speed (1/250s) to capture detail without motion blur,
		low ISO (100) for minimal noise in bright conditions,
		and a narrower aperture (f/11) for a sharper focus across the landscape.
		*/
		cam->setAperture(11.0f);
		cam->setShutterSecs(0.004f);//1/250s
		cam->setISO(100);
	}

	//Indoor Portrait preset
	inline void applyIndoorPortraitCameraPreset(CameraController* cam)
	{
		/*
		Slightly slower shutter speed (1/125s) to allow more light for indoor settings,
		higher ISO (400) to compensate for lower light levels without introducing too much noise,
		and a wider aperture (f/2.8) for a soft background blur while keeping the subject in focus.
		*/
		cam->setAperture(2.8f);
		cam->setShutterSecs(0.008f);//1/125s
		cam->setISO(400);
	}

	//NightTime Cityscape preset
	inline void applyNightTimeCityscapeCameraPreset(CameraController* cam)
	{
		/*
		Longer exposure (2s) to capture light in low-light conditions,
		moderate ISO (800) for sensitivity without excessive noise,
		and a moderate aperture (f/4) to balance depth of field and light intake.
		*/
		cam->setAperture(4.0f);
		cam->setShutterSecs(0.05f);//2s
		cam->setISO(800);
	}
}