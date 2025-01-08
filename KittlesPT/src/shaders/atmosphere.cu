#include "atmosphere.cuh"
#include "ray.cuh"

namespace KittlesPT
{
	__device__ float angularDiameterToPhysicalDiameter(float angle_rad, float distance)
	{
		return 2.0f * distance * tanf(angle_rad / 2.0f);
	}

	__device__ float3 sphericalToSunDirection(float theta, float phi)
	{
		return normalize(make_float3(
			cosf(phi) * cosf(theta),
			sinf(theta),
			sinf(phi) * cosf(theta)
		));
	}

	__device__ bool intersectSphere(const Ray& ray, float3 t_sphere_centre, float t_sphere_radius, float& r_t0, float& r_t1)
	{
		float a = dot(ray.getDirection(), ray.getDirection()); //TODO: just length()
		float b = -2.0f * dot(ray.getDirection(), t_sphere_centre - ray.getOrigin());
		float c = dot(t_sphere_centre - ray.getOrigin(), t_sphere_centre - ray.getOrigin()) - Sqr(t_sphere_radius);

		float discriminant = Sqr(b) - 4 * a * c;

		if (discriminant < 0.0f)
		{
			//r_t1 = -1.0f; // Set r_t1 to -1 to indicate a miss
			return false;
		}

		//(discriminant == 0.0f) = tangent

		float sqrtDiscriminant = sqrtf(discriminant);
		r_t0 = (-b - sqrtDiscriminant) / (2.0f * a);
		r_t1 = (-b + sqrtDiscriminant) / (2.0f * a);//bigger

		// make r_t0 is the smaller value
		if (r_t0 > r_t1)
		{
			cuda::std::swap(r_t0, r_t1);
		}

		return true;
	}

	__device__ RGBSpectrum Atmosphere::sampleLe(float3 t_orig, float3 t_dir, float t_tmin, float t_tmax) const
	{
		float t0, t1;
		// miss atmosphere
		if (!intersectSphere(Ray(t_orig, t_dir), make_float3(0), m_atmosphere_radius, t0, t1) || t1 < 0)
		{
			return RGBSpectrum(0);
		}
		// hit atmosphere
		if (t0 > t_tmin && t0 > 0) {
			t_tmin = t0; // increase tmin
		}
		if (t1 < t_tmax) {
			t_tmax = t1; // reduce tmax
		}

		const float step_length = (t_tmax - t_tmin) / m_num_samples;
		float current_t = t_tmin;
		RGBSpectrum sum_R_transmission = RGBSpectrum(0);
		RGBSpectrum sum_M_transmission = RGBSpectrum(0);        // integrated mie and rayleigh contribution
		float sum_R_optical_depth = 0, sum_M_optical_depth = 0; // discrete integration for transmittance

		const float mu = dot(t_dir, m_sun_direction); // cosine of the angle between the sun direction and the ray direction
		const float phaseR = 3.f / (16.f * Constants::PI) * (1 + mu * mu);
		const float g = 0.76f; // anisotropy
		const float phaseM = 3.f / (8.f * Constants::PI) * ((1.f - g * g) * (1.f + mu * mu)) / ((2.f + g * g) * pow(1.f + g * g - 2.f * g * mu, 1.5f));

		// sample contrib points along ray;
		// integrate from t0 to t1 for transmittance and Lsun
		for (uint32_t i = 0; i < m_num_samples; ++i)
		{
			const float3 sample_position = t_orig + (current_t + step_length * 0.5f) * t_dir;
			const float height = length(sample_position) - m_earth_radius;

			// compute optical depth for this step
			const float hr = exp(-height / Hr) * step_length;
			const float hm = exp(-height / Hm) * step_length;
			sum_R_optical_depth += hr;
			sum_M_optical_depth += hm;

			// optical depth sum for light for this step
			float t0_light, t1_light;
			intersectSphere(Ray(sample_position, m_sun_direction), make_float3(0),
				m_atmosphere_radius, t0_light, t1_light);
			const float step_length_light = t1_light / m_num_samples_light;

			float current_t_light = 0;
			float sum_R_optical_depth_light = 0,
				sum_M_optical_depth_light = 0;

			uint32_t j;
			for (j = 0; j < m_num_samples_light; ++j)
			{
				const float3 sample_position_light = sample_position + (current_t_light + step_length_light * 0.5f) * m_sun_direction;
				const float height_light = length(sample_position_light) - m_earth_radius;
				if (height_light < 0) // if sun dir points/sample_pos is below horizon/earth
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

				const float& optical_depthR = hr, optical_depthM = hm;
				// transmittance * optical_depth sum; later multiplied with beta to compute beta(h)=beta(0)*optical_depth(h)
				sum_R_transmission += transmittance * optical_depthR;
				sum_M_transmission += transmittance * optical_depthM;
			}
			current_t += step_length;
		}

		RGBSpectrum col = (sum_R_transmission * betaR_scattering_coeff * phaseR + sum_M_transmission * betaM_scattering_coeff * phaseM) * m_sun_intensity;
		//float3 col = (sum_R_transmission * betaR_scattering_coeff * phaseR) * m_sun_intensity;//rayleigh only
		//float3 col = (sum_M_transmission * betaM_scattering_coeff * phaseM) * m_sun_intensity; //mie only
		return col;
	}
}