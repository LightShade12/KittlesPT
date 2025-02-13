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

	inline __device__ bool intersectSphere(const Ray& ray, float3 p_sphere_centre, float p_sphere_radius, float* r_t0, float* r_t1)
	{
		float3 oc = p_sphere_centre - ray.getOrigin();
		float a = dot(ray.getDirection(), ray.getDirection());
		float b = -2.0f * dot(ray.getDirection(), oc);
		float c = dot(oc, oc) - Sqr(p_sphere_radius);

		float discriminant = Sqr(b) - 4.0f * a * c;

		if (discriminant < 0.0f) {
			return false;
		}

		float sqrt_discriminant = sqrtf(discriminant);
		*(r_t0) = (-b - sqrt_discriminant) / (2.0f * a);
		*(r_t1) = (-b + sqrt_discriminant) / (2.0f * a);

		if (*(r_t0) > *(r_t1)) {
			cuda::std::swap(*(r_t0), *(r_t1));
		}
		//sphere behind ray
		if (*(r_t1) < 0.0f) {
			return false;
		}

		return true;
	};

	namespace Atmosphere
	{
		namespace Values {
			//IAU 2012 Resolution B2
			__constant__ constexpr float SUN_PHYSICAL_DISTANCE_METERS = 149.597e9f;//149 597 870 700 m
			//PBR BOOK v4/Radiometry/Table 4.1
			__constant__ constexpr float SUN_HORIZON_LUMINANCE_NITS = 6.0e5f;
		}

		inline __device__ float angularDiameterToPhysicalDiameter(float angle_rad, float distance)
		{
			return 2.0f * distance * tanf(angle_rad / 2.0f);
		};

		//D65 = 6504K
		class NishitaAtmosphereModel
		{
		public:

			__device__ NishitaAtmosphereModel(const float3& p_sun_direction, float p_sun_emission_factor = 1.0f) :
				m_sun_direction(p_sun_direction),
				m_sun_emission_factor(p_sun_emission_factor)
			{};

			__device__ NishitaAtmosphereModel(float p_sun_phi, float p_sun_theta, float p_sun_emission_factor = 1.0f) :
				m_sun_emission_factor(p_sun_emission_factor)
			{
				m_sun_direction = normalize(make_float3(
					cosf(p_sun_phi) * cosf(p_sun_theta),
					sinf(p_sun_theta),
					sinf(p_sun_phi) * cosf(p_sun_theta)
				));
			};

			__device__ RGBSpectrum sampleLe(const Ray& ray) const
			{
				//determine ray-volume configuration
				float t_tmin = 0.0f, t_tmax = INFINITY;
				{
					float t_enter, t_exit;
					// miss atmosphere
					if (!intersectSphere(ray, make_float3(0.0f), m_atmosphere_radius_meters, &t_enter, &t_exit) || t_exit < 0.0f) {
						return RGBSpectrum(0.0f);
					}
					// hit atmosphere
					if (t_enter > t_tmin && t_enter > 0.0f) {
						t_tmin = t_enter; // increase tmin
					}
					if (t_exit < t_tmax) {
						t_tmax = t_exit; // reduce tmax
					}
				}
				//-------

				const float volume_depth = t_tmax - t_tmin;
				const float delta_t = volume_depth / m_raymarch_steps_num;
				float current_t = t_tmin;
				RGBSpectrum integration_R_Tr(0.0f), integration_M_Tr(0.0f);
				float integration_partial_sigma_t_R = 0.0f, integration_partial_sigma_t_M = 0.0f;

				// sample contrib points along ray;
				// integrate from t_enter to t_exit for transmittance and Lsun
#pragma unroll
				for (uint32_t i = 0; i < m_raymarch_steps_num; ++i)
				{
					const float3 sample_position = ray.getPointAt(current_t + (delta_t * 0.5f));
					const float sampling_altitude = length(sample_position) - m_earth_radius_meters;

					// compute tau for this step
					const float partial_sigma_t_R = ::expf(-sampling_altitude / Hr) * delta_t;//the varying term of sigma_t() calculation
					const float partial_sigma_t_M = ::expf(-sampling_altitude / Hm) * delta_t;
					integration_partial_sigma_t_R += partial_sigma_t_R;//integrating sigma_t for computing Tr by intergrating varying term
					integration_partial_sigma_t_M += partial_sigma_t_M;

					float t_enter_Li, t_exit_Li;
					Ray ray_Li = Ray(sample_position, m_sun_direction);
					intersectSphere(ray_Li, make_float3(0.0f),
						m_atmosphere_radius_meters, &t_enter_Li, &t_exit_Li);
					const float delta_t_Li = t_exit_Li / m_raymarch_Li_steps_num;

					float current_t_Li = 0.0f;
					// tau sum for Li for current step
					float integration_partial_sigma_t_R_Li = 0.0f, integration_partial_sigma_t_M_Li = 0.0f;

					//Single scattering in-scattering
					uint8_t j;
#pragma unroll
					for (j = 0; j < m_raymarch_Li_steps_num; ++j)
					{
						const float3 sample_position_Li = ray_Li.getPointAt(current_t_Li + (delta_t_Li * 0.5f));
						const float altitude_Li = length(sample_position_Li) - m_earth_radius_meters;
						if (altitude_Li < 0.0f) // if sun dir points/sample_pos is below horizon/earth
						{
							break;
						}
						integration_partial_sigma_t_R_Li += ::expf(-altitude_Li / Hr) * delta_t_Li;//integrating partial sigma_t
						integration_partial_sigma_t_M_Li += ::expf(-altitude_Li / Hm) * delta_t_Li;
						current_t_Li += delta_t_Li;
					}
					if (j == m_raymarch_Li_steps_num) // last iter
					{
						const RGBSpectrum sigma_t_R_sea_level = betaR_sea_level_scattering_coeff + sigma_a_R;
						const RGBSpectrum sigma_t_M_sea_level = betaM_sea_level_scattering_coeff * 1.1f;

						//integrated tau
						const RGBSpectrum tau = sigma_t_R_sea_level * (integration_partial_sigma_t_R + integration_partial_sigma_t_R_Li)
							+ sigma_t_M_sea_level * (integration_partial_sigma_t_M + integration_partial_sigma_t_M_Li);

						const RGBSpectrum Tr = exp(-tau);

						const float optical_depthR = partial_sigma_t_R, optical_depthM = partial_sigma_t_M;
						// transmittance * optical_depth sum; later multiplied with beta to compute beta(h)=beta(0)*optical_depth(h)
						integration_R_Tr += Tr * optical_depthR;
						integration_M_Tr += Tr * optical_depthM;
					}
					current_t += delta_t;
				}

				const float mu = dot(ray.getDirection(), m_sun_direction);

				const RGBSpectrum atmosphere_radiance =
					(integration_R_Tr * betaR_sea_level_scattering_coeff * rayleighPhaseFunction(mu)
						+ integration_M_Tr * betaM_sea_level_scattering_coeff * miePhaseFunction(mu));

				return atmosphere_radiance * Values::SUN_HORIZON_LUMINANCE_NITS * m_sun_emission_factor;
			}

			__device__ float getAtmosphereLuminance() {
				return m_sun_emission_factor;
			}

			__device__ float getEarthRadiusMeters() const {
				return m_earth_radius_meters;
			}

			__device__ float3 getSunDirection() const {
				return m_sun_direction;
			}

		private:
			//phase func Rayleigh
			__device__ float rayleighPhaseFunction(float cos_theta) const
			{
				const float numerator = 3.0f * (1.0f + Sqr(cos_theta));
				const float probability = numerator / (16.0f * Constants::PI);
				return probability;
			}

			//phase func Mie; variant of HGPhaseFunction; forward scattering in nature
			__device__ float miePhaseFunction(float cos_theta) const
			{
				constexpr float g = 0.76f; // anisotropy
				const float denom = (8.0f * Constants::PI) * ((2.0f + Sqr(g)) * pow(1.0f + Sqr(g) - 2.0f * g * cos_theta, 3.0f / 2.0f));
				const float probability = 3.0f * ((1.0f - Sqr(g)) * (1.0f + Sqr(cos_theta))) / denom;
				return probability;
			}

			uint32_t m_raymarch_steps_num = 16u;//ray-march samples
			uint32_t m_raymarch_Li_steps_num = 8u;//sun in-scattering samples

			float3 m_sun_direction = make_float3(0.0f, 0.0f, 0.0f);
			float m_sun_emission_factor = 1.0f;

			float m_earth_radius_meters = 6360e3f;      // In the paper: Rg or Re (radius ground, eart)
			//60km (2 layers: tropo/stratosphere) of atmosphere simulated
			float m_atmosphere_radius_meters = 6420e3f; // In the paper: R or Ra (radius atmosphere)
			//H, scale height; temperature dependant
			float Hr = 7994.0f;                   // Thickness of the atmosphere if density was uniform (Hr)
			float Hm = 1200.0f;                   // Same as above but for Mie scattering (Hm)

			//single scattering albedo or "in/out-scattering coefficient"; sigma_s_rayleigh
			const RGBSpectrum betaR_sea_level_scattering_coeff = RGBSpectrum(3.8e-6f, 13.5e-6f, 33.1e-6f);//for nano-particles; precomputed for sea-level
			const RGBSpectrum betaM_sea_level_scattering_coeff = RGBSpectrum(21e-6f);//for micro particles; precomputed for sea-level

			//Absorption coefficient is zero
			const float sigma_a_R = 0.0f;
		};
	}/*Atmosphere*/
}/*KittlesPT*/