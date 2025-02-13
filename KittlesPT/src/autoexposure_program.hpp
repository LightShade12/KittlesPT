#pragma once
#include "maths/linear_algebra.cuh"
#include "pod_types.hpp"

namespace KittlesPT
{
	class AutoExposureProgram
	{
	public:
		void recordValues(const ExposureValues& exposure_values, float ev_comp, float white_point_ev, float black_point_ev)
		{
			m_exposure_values = exposure_values;
			m_ev_comp = ev_comp;
			m_white_point = white_point_ev;
			m_black_point = black_point_ev;
		};

		void computeExposure(float average_scene_luminance)
		{
			float ev_target = computeEV100(average_scene_luminance) - m_ev_comp;
			ev_target += 6.0f;//TODO:FIXME:Nasty fix for auto exposure undererestimation
			applyAperturePriority(0.1f, ev_target,
				m_exposure_values.aperture_f_num, m_exposure_values.shutter_speed_secs, m_exposure_values.ISO);
		};

		static float getSaturationBasedExposure(float aperture, float shutter_time, float iso)
		{
			//measuring for iso = S pmax
			constexpr float q = 0.65f;
			float l_max = (78.0f / q) * (Sqr(aperture) / (iso * shutter_time));
			return 1.0f / l_max;//why reciprocal?
		}

		static float getStandardOutputBasedExposure(float aperture,
			float shutterSpeed,
			float iso,
			float middleGrey = 0.18f)
		{
			//for 18% gray derived from 118/255 after gamma correction
			constexpr float q = 0.65f;
			float l_avg = (10.0f / q) * (Sqr(aperture) / (iso * shutterSpeed));
			return middleGrey / l_avg;
		}

		ExposureValues getExposureValues() const { return m_exposure_values; };
		float getEVComp() const { return m_ev_comp; }
		float getWhitePoint() const { return m_white_point; }
		float getBlackPoint() const { return m_black_point; }

	private:
		ExposureValues m_exposure_values;
		float m_ev_comp = 0.0f;
		float m_white_point = 0.0f;
		float m_black_point = 0.0f;

		float computeEV100(float average_luminance)
		{
			// K is a light meter calibration constant
			constexpr float K = 12.5f;
			return log2((average_luminance * 100.0f) / K);
		}

		// Notes:
		// EV below refers to EV at ISO 100

		// Given an aperture, shutter speed, and exposure value compute the required ISO value
		float computeISO(float aperture, float shutterSpeed, float ev)
		{
			return (Sqr(aperture) * 100.0f) / (shutterSpeed * pow(2.0f, ev));
		}

		// Given the camera settings compute the current exposure value
		float computeEV(float aperture, float shutterSpeed, float iso)
		{
			return log2((Sqr(aperture) * 100.0f) / (shutterSpeed * iso));
		}

		void applyAperturePriority(float focalLength,
			float targetEV,
			float& aperture,
			float& shutterSpeed,
			float& iso)
		{
			// Start with the assumption that we want a shutter speed of 1/f
			shutterSpeed = 1.0f / (focalLength * 1000.0f);
			// Compute the resulting ISO if we left the shutter speed here
			iso = clamp(computeISO(aperture, shutterSpeed, targetEV),
				static_cast<float>(m_exposure_values.MIN_ISO), static_cast<float>(m_exposure_values.MAX_ISO));
			// Figure out how far we were from the target exposure value
			float evDiff = targetEV - computeEV(aperture, shutterSpeed, iso);
			// Compute the final shutter speed
			shutterSpeed = clamp(shutterSpeed * pow(2.0f, -evDiff),
				m_exposure_values.MIN_SHUTTER_SECS, m_exposure_values.MAX_SHUTTER_SECS);
		}
	};
}