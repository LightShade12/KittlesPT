#pragma once
#include "color.cuh"
#include <vector_types.h>

namespace KittlesPT
{
	class Ray;

	__constant__ constexpr float SUN_VISIBILITY_DISTANCE_METERS = 100.0f;

	__device__ float angularDiameterToPhysicalDiameter(float angle_rad, float distance);

	__device__ float3 sphericalToSunDirection(float theta, float phi);

	class Atmosphere
	{
	public:

		__device__ Atmosphere(float3 t_sun_direction, float t_sun_intensity) :
			m_sun_direction(t_sun_direction),
			m_sun_intensity(t_sun_intensity) {};

		__device__ RGBSpectrum sampleLe(float3 t_orig, float3 t_dir, float t_tmin, float t_tmax) const;

		__device__ float getEarthRadius() const {
			return m_earth_radius;
		}

		__device__ float3 getSunDirection() const {
			return m_sun_direction;
		}

	private:

		uint32_t m_num_samples = 16u;
		uint32_t m_num_samples_light = 8u;

		float m_sun_intensity = 1.0f;
		float3 m_sun_direction{ 0.0f,0.0f,0.0f };// The sun direction (normalized)

		float m_earth_radius = 6360e3f;      // In the paper this is usually Rg or Re (radius ground, eart)
		float m_atmosphere_radius = 6420e3f; // In the paper this is usually R or Ra (radius atmosphere)
		float Hr = 7994.0f;                   // Thickness of the atmosphere if density was uniform (Hr)
		float Hm = 1200.0f;                   // Same as above but for Mie scattering (Hm)

		const RGBSpectrum betaR_scattering_coeff = RGBSpectrum(3.8e-6f, 13.5e-6f, 33.1e-6f);
		const RGBSpectrum betaM_scattering_coeff = RGBSpectrum(21e-6f);
	};
}/*KittlesPT*/