#pragma once
#include "ray.cuh"
#include "maths/linear_algebra.cuh"
#include "maths/constants.cuh"
#include "color.cuh"
#include <cuda/std/span>
#include <vector_types.h>

namespace KittlesPT
{
	//TODO: add units

	//IAU 2012 Resolution B2
	__constant__ constexpr float SUN_PHYSICAL_DISTANCE_METERS = 149.597e9f;

	__constant__ constexpr float SUN_VISIBILITY_TEST_DISTANCE_METERS = 100.0f;

	inline __device__ float angularDiameterToPhysicalDiameter(float angle_rad, float distance)
	{
		return 2.0f * distance * tanf(angle_rad / 2.0f);
	};

	inline __device__ bool intersectSphere(const Ray& ray, float3 t_sphere_centre, float t_sphere_radius, float& r_t0, float& r_t1)
	{
		float3 oc = t_sphere_centre - ray.getOrigin();
		float a = dot(ray.getDirection(), ray.getDirection());
		float b = -2.0f * dot(ray.getDirection(), oc);
		float c = dot(oc, oc) - Sqr(t_sphere_radius);

		float discriminant = Sqr(b) - 4.0f * a * c;

		if (discriminant < 0.0f)
		{
			return false;
		}

		float sqrtDiscriminant = sqrtf(discriminant);
		r_t0 = (-b - sqrtDiscriminant) / (2.0f * a);
		r_t1 = (-b + sqrtDiscriminant) / (2.0f * a);

		if (r_t0 > r_t1)
		{
			cuda::std::swap(r_t0, r_t1);
		}
		//sphere behind ray
		if (r_t1 < 0.0f)
		{
			return false;
		}

		return true;
	};

	class Atmosphere
	{
	public:

		__device__ Atmosphere(float3 t_sun_direction, float t_sun_emission_nits) :
			m_sun_direction(t_sun_direction),
			m_sun_emission_scale_nits(t_sun_emission_nits) {};

		__device__ RGBSpectrum sampleLe(Ray ray, float t_tmin = 0.0f, float t_tmax = INFINITY) const
		{
			float t0, t1;
			// miss atmosphere
			if (!intersectSphere(ray, make_float3(0.0f), m_atmosphere_radius_meters, t0, t1) || t1 < 0.0f) {
				return RGBSpectrum(0.0f);
			}
			// hit atmosphere
			if (t0 > t_tmin && t0 > 0.0f) {
				t_tmin = t0; // increase tmin
			}
			if (t1 < t_tmax) {
				t_tmax = t1; // reduce tmax
			}

			const float step_length = (t_tmax - t_tmin) / m_num_samples;
			float current_t = t_tmin;
			RGBSpectrum sum_R_transmission = RGBSpectrum(0.0f);
			RGBSpectrum sum_M_transmission = RGBSpectrum(0.0f);     // integrated mie and rayleigh contribution
			float sum_R_optical_depth = 0.0f, sum_M_optical_depth = 0.0f; // discrete integration for transmittance

			// sample contrib points along ray;
			// integrate from t0 to t1 for transmittance and Lsun
#pragma unroll
			for (uint32_t i = 0; i < m_num_samples; ++i)
			{
				const float3 sample_position = ray.getPointAt(current_t + (step_length * 0.5f));
				const float height = length(sample_position) - m_earth_radius_meters;

				// compute optical depth for this step
				const float hr = exp(-height / Hr) * step_length;
				const float hm = exp(-height / Hm) * step_length;
				sum_R_optical_depth += hr;
				sum_M_optical_depth += hm;

				// optical depth sum for light for current step
				float t0_light, t1_light;
				intersectSphere(Ray(sample_position, m_sun_direction), make_float3(0.0f),
					m_atmosphere_radius_meters, t0_light, t1_light);
				const float step_length_light = t1_light / m_num_samples_light;

				float current_t_light = 0.0f;
				float sum_R_optical_depth_light = 0.0f,
					sum_M_optical_depth_light = 0.0f;

				uint8_t j;
#pragma unroll
				for (j = 0; j < m_num_samples_light; ++j)
				{
					const float3 sample_position_light = sample_position + (current_t_light + step_length_light * 0.5f) * m_sun_direction;
					const float height_light = length(sample_position_light) - m_earth_radius_meters;
					if (height_light < 0.0f) // if sun dir points/sample_pos is below horizon/earth
					{
						break;
					}
					sum_R_optical_depth_light += exp(-height_light / Hr) * step_length_light;
					sum_M_optical_depth_light += exp(-height_light / Hm) * step_length_light;
					current_t_light += step_length_light;
				}
				if (j == m_num_samples_light) // last iter
				{
					RGBSpectrum beta_R_extinction = betaR_scattering_coeff;
					RGBSpectrum beta_M_extinction = betaM_scattering_coeff * 1.1f;

					// transmittance; grouping optical_depth sum for sunlight and view transmittance calcs
					RGBSpectrum tau = beta_R_extinction * (sum_R_optical_depth + sum_R_optical_depth_light) + beta_M_extinction * (sum_M_optical_depth + sum_M_optical_depth_light);
					RGBSpectrum transmittance = RGBSpectrum(exp(-tau.r), exp(-tau.g), exp(-tau.b));

					const float optical_depthR = hr, optical_depthM = hm;
					// transmittance * optical_depth sum; later multiplied with beta to compute beta(h)=beta(0)*optical_depth(h)
					sum_R_transmission += transmittance * optical_depthR;
					sum_M_transmission += transmittance * optical_depthM;
				}
				current_t += step_length;
			}

			const float mu = dot(ray.getDirection(), m_sun_direction);
			RGBSpectrum atmosphere_radiance = (sum_R_transmission * betaR_scattering_coeff * phaseR(mu)
				+ sum_M_transmission * betaM_scattering_coeff * phaseM(mu)) * m_sun_emission_scale_nits;
			//float3 atmosphere_radiance = (sum_R_transmission * betaR_scattering_coeff * phaseR) * m_sun_emission_scale_nits;//rayleigh only
			//float3 atmosphere_radiance = (sum_M_transmission * betaM_scattering_coeff * phaseM) * m_sun_emission_scale_nits; //mie only
			return atmosphere_radiance;
		}

		__device__ float getEarthRadiusMeters() const {
			return m_earth_radius_meters;
		}

		__device__ float3 getSunDirection() const {
			return m_sun_direction;
		}

	private:
		//phase func Rayleigh
		__device__ float phaseR(float cos_theta) const
		{
			const float phaseR = 3.0f / (16.0f * Constants::PI) * (1.0f + Sqr(cos_theta));
			return phaseR;
		}
		//phase func Mie; variant of HGPhaseFunction; forward scattering in nature
		__device__ float phaseM(float cos_theta) const
		{
			constexpr float g = 0.76f; // anisotropy
			const float phaseM = 3.0f / (8.0f * Constants::PI) * ((1.0f - Sqr(g)) * (1.0f + Sqr(cos_theta))) / ((2.0f + Sqr(g)) * pow(1.0f + Sqr(g) - 2.0f * g * cos_theta, 1.5f));
			return phaseM;
		}

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