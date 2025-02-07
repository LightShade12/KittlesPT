#include "atmosphere.cuh"
#include "ray.cuh"
#include "maths/linear_algebra.cuh"
#include "maths/constants.cuh"
#include <cuda/std/span>

namespace KittlesPT
{
	__device__ float angularDiameterToPhysicalDiameter(float angle_rad, float distance)
	{
		return 2.0f * distance * tanf(angle_rad / 2.0f);
	}

	__device__ bool intersectSphere(const Ray& ray, float3 t_sphere_centre, float t_sphere_radius, float& r_t0, float& r_t1)
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
	}

	__device__ float Atmosphere::phaseR(float mu) const
	{
		const float phaseR = 3.0f / (16.0f * Constants::PI) * (1.0f + mu * mu);
		return phaseR;
	}

	__device__ float Atmosphere::phaseM(float mu) const
	{
		constexpr float g = 0.76f; // anisotropy
		const float phaseM = 3.0f / (8.0f * Constants::PI) * ((1.0f - g * g) * (1.0f + mu * mu)) / ((2.0f + g * g) * pow(1.0f + g * g - 2.0f * g * mu, 1.5f));
		return phaseM;
	}

	__device__ RGBSpectrum Atmosphere::sampleLe(Ray ray, float t_tmin, float t_tmax) const
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
}/*KittlesPT*/