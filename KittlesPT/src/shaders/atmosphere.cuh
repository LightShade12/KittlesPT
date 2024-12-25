#pragma once
#include "maths/linear_algebra.cuh"
#include "maths/constants.cuh"
#include "color.cuh"
#include <cuda/std/span>

namespace KittlesPT
{
	__device__ bool intersectSphere(float3 t_orig, float3 t_dir, float3 t_centre, float t_radius, float& r_t0, float& r_t1);

	__constant__ constexpr float SUN_DISTANCE_METERS = 100.0f;

	__device__ float AngularDiameterToPhysicalDiameter(float angle_rad, float distance);

	struct Atmosphere
	{
		__device__ Atmosphere(float3 t_sunpos, float t_sun_intensity) :
			m_sun_position(t_sunpos), m_sun_intensity(t_sun_intensity) {};

	public:

		__device__ RGBSpectrum Le(float3 t_orig, float3 t_dir, float t_tmin, float t_tmax) const;

	public:

		uint32_t m_num_samples = 16;
		uint32_t m_num_samples_light = 8;

		float m_sun_intensity = 1;
		float3 m_sun_position;             // The sun direction (normalized)
		float m_earth_radius = 6360e3;      // In the paper this is usually Rg or Re (radius ground, eart)
		float m_atmosphere_radius = 6420e3; // In the paper this is usually R or Ra (radius atmosphere)
		float Hr = 7994;                   // Thickness of the atmosphere if density was uniform (Hr)
		float Hm = 1200;                   // Same as above but for Mie scattering (Hm)

		const RGBSpectrum betaR_scattering_coeff = RGBSpectrum(3.8e-6f, 13.5e-6f, 33.1e-6f);
		const RGBSpectrum betaM_scattering_coeff = RGBSpectrum(21e-6f);
	};
}