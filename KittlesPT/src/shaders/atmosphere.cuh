#pragma once
#include "color.cuh"
#include <vector_types.h>

namespace KittlesPT
{
	//TODO: add units
	class Ray;

	//IAU 2012 Resolution B2
	__constant__ constexpr float SUN_PHYSICAL_DISTANCE_METERS = 149.597e9f;

	__constant__ constexpr float SUN_VISIBILITY_TEST_DISTANCE_METERS = 100.0f;

	__device__ float angularDiameterToPhysicalDiameter(float angle_rad, float distance);

	class Atmosphere
	{
	public:

		__device__ Atmosphere(float3 t_sun_direction, float t_sun_emission_nits) :
			m_sun_direction(t_sun_direction),
			m_sun_emission_scale_nits(t_sun_emission_nits) {};

		__device__ RGBSpectrum sampleLe(Ray ray, float t_tmin = 0.0f, float t_tmax = INFINITY) const;

		__device__ float getEarthRadiusMeters() const {
			return m_earth_radius_meters;
		}

		__device__ float3 getSunDirection() const {
			return m_sun_direction;
		}

	private:
		__device__ float phaseR(float mu) const;//phase func Rayleigh
		__device__ float phaseM(float mu) const;//phase func Mie

		uint32_t m_num_samples = 16u;//ray-march samples
		uint32_t m_num_samples_light = 8u;//sun in-scattering samples

		float m_sun_emission_scale_nits = 1.0f;
		float3 m_sun_direction{ 0.0f,0.0f,0.0f };// The sun direction (normalized)

		float m_earth_radius_meters = 6360e3f;      // In the paper: Rg or Re (radius ground, eart)
		float m_atmosphere_radius_meters = 6420e3f; // In the paper: R or Ra (radius atmosphere)
		float Hr = 7994.0f;                   // Thickness of the atmosphere if density was uniform (Hr)
		float Hm = 1200.0f;                   // Same as above but for Mie scattering (Hm)

		const RGBSpectrum betaR_scattering_coeff = RGBSpectrum(3.8e-6f, 13.5e-6f, 33.1e-6f);
		const RGBSpectrum betaM_scattering_coeff = RGBSpectrum(21e-6f);
	};
}/*KittlesPT*/