// ----------------------------------------------------------------------------
// atmosphere.cuh
//
// Implements the Nishita atmospheric scattering model for simulating
// Earth's atmosphere in a single-scattering approximation.
// References: https://www.scratchapixel.com/lessons/procedural-generation-virtual-worlds/simulating-sky/simulating-colors-of-the-sky.html
// https://en.wikipedia.org/wiki/IAU_(1976)_System_of_Astronomical_Constants
// ----------------------------------------------------------------------------

#pragma once
#include "ray.cuh"
#include "maths/linear_algebra.cuh"
#include "maths/constants.cuh"
#include "color.cuh"

#include <cuda/std/span>
#include <vector_types.h>

namespace KittlesPT
{
	/// <summary>
	/// Inline ray-sphere intersection testing utility
	/// </summary>
	/// <param name="p_ray">Ray</param>
	/// <param name="p_sphere_centre">sphere's position</param>
	/// <param name="p_sphere_radius">sphere's radius</param>
	/// <param name="r_t0"> return value for entry distance</param>
	/// <param name="r_t1"> return value for exit distance</param>
	/// <returns></returns>
	inline __device__ bool intersectSphere(const Ray& p_ray, float3 p_sphere_centre, float p_sphere_radius,
		float* r_t0, float* r_t1)
	{
		float3 oc = p_sphere_centre - p_ray.getOrigin();
		float a = dot(p_ray.getDirection(), p_ray.getDirection());
		float b = -2.0f * dot(p_ray.getDirection(), oc);
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
		/**
		 * @namespace Values
		 * @brief Contains constant values used in the Atmosphere namespace,
		 *        primarily for physical and rendering parameters.
		 */
		namespace Values {
			// IAU 2012 Resolution B2 -  Physical distance from Earth to Sun (1 AU)
			__constant__ constexpr float SOLAR_DISTANCE_EARTH_METERS = 149.597e9f;//149 597 870 700 metres
			// PBR BOOK v4/Radiometry/Table 4.1 - Luminance of the Sun at the horizon, as seen from Earth
			__constant__ constexpr float SUN_HORIZON_LUMINANCE_NITS = 6.0e5f; // Nits (candelas per square meter)
		}

		/**
		 * @brief Converts an angular diameter (in radians) to a physical diameter at a given distance.
		 *
		 * @param angle_rad Angular diameter in radians.
		 * @param distance Distance to the object.
		 * @return Physical diameter of the object at the given distance.
		 */
		inline __device__ float angularDiameterToPhysicalDiameter(float angle_rad, float distance)
		{
			return 2.0f * distance * tanf(angle_rad / 2.0f);
		}

		/**
		 * @class NishitaAtmosphereModel
		 * @brief Implements the Nishita atmosphere rendering model for single scattering.
		 *
		 * This model simulates the sky color and atmospheric scattering effects
		 * based on the Nishita et al. paper. It accounts for Rayleigh and Mie scattering.
		 *
		 * @author [Your Name/Organization] (if applicable)
		 * @date [Date of Creation/Modification]
		 * @version 1.0
		 */
		class NishitaAtmosphereModel
		{
		public:
			/**
			 * @brief Constructor for NishitaAtmosphereModel, specifying sun direction as a normalized vector.
			 *
			 * @param p_sun_direction Normalized direction vector from the origin to the sun.
			 * @param p_sun_emission_factor Scaling factor for sun's intensity. Default is 1.0f.
			 */
			__device__ NishitaAtmosphereModel(const float3& p_sun_direction, float p_sun_emission_factor = 1.0f) :
				m_sunDirection(p_sun_direction),
				m_sunEmissionFactor(p_sun_emission_factor)
			{};

			/**
			 * @brief Constructor for NishitaAtmosphereModel, specifying sun direction using spherical coordinates.
			 *
			 * @param p_sun_phi Azimuthal angle (phi) in radians.
			 * @param p_sun_theta Polar angle (theta) in radians (from the Y-axis).
			 * @param p_sun_emission_factor Scaling factor for sun's intensity. Default is 1.0f.
			 */
			__device__ NishitaAtmosphereModel(float p_sun_phi, float p_sun_theta, float p_sun_emission_factor = 1.0f) :
				m_sunEmissionFactor(p_sun_emission_factor)
			{
				m_sunDirection = normalize(make_float3(
					cosf(p_sun_phi) * cosf(p_sun_theta),
					sinf(p_sun_theta),
					sinf(p_sun_phi) * cosf(p_sun_theta)
				));
			};

			/**
			 * @brief Samples the in-scattered radiance (Li) along a ray due to atmospheric scattering.
			 *
			 * @param p_ray The ray for which to sample the in-scattered radiance.
			 * @return RGBSpectrum representing the in-scattered radiance. Returns black (0.0f) if the ray does not intersect the atmosphere.
			 */
			__device__ RGBSpectrum sampleLi(const Ray& p_ray) const
			{
				// 1. Determine ray-atmosphere volume intersection
				float ray_t_min = 0.0f, ray_t_max = INFINITY;
				{
					float intersection_t_enter, intersection_t_exit;
					// Miss atmosphere
					if (!intersectSphere(p_ray, make_float3(0.0f), m_atmosphereRadiusMeters,
						&intersection_t_enter, &intersection_t_exit) || intersection_t_exit < 0.0f) {
						return RGBSpectrum(0.0f); // No atmosphere intersection, return black
					}
					// Hit atmosphere
					if (intersection_t_enter > ray_t_min && intersection_t_enter > 0.0f) {
						ray_t_min = intersection_t_enter; // Increase ray start to atmosphere entry
					}
					if (intersection_t_exit < ray_t_max) {
						ray_t_max = intersection_t_exit;   // Reduce ray end to atmosphere exit
					}
				}
				//------- Ray-volume configuration now defined by ray_t_min and ray_t_max

				const float volume_depth = ray_t_max - ray_t_min;
				const float ray_step_size = volume_depth / rayMarchingSteps;
				float current_ray_t = ray_t_min;
				RGBSpectrum integration_R_Tr(0.0f), integration_M_Tr(0.0f); // Integrated terms for Rayleigh and Mie scattering
				float integration_partial_sigma_t_R = 0.0f, integration_partial_sigma_t_M = 0.0f; // Integrated partial optical depths

				// 2. Raymarch along the view ray and integrate scattering
#pragma unroll // Hint to the compiler to unroll this loop for performance (CUDA context)
				for (uint32_t i = 0; i < rayMarchingSteps; ++i)
				{
					const float3 sample_position = p_ray.getPointAt(current_ray_t + (ray_step_size * 0.5f)); // Midpoint sampling
					const float sampling_altitude = length(sample_position) - m_earthRadiusMeters; // Altitude above Earth's surface

					// Compute partial optical depth (tau) for this ray step, based on altitude-dependent density
					const float partial_sigma_t_R = ::expf(-sampling_altitude / m_Hr) * ray_step_size; // Rayleigh varying term
					const float partial_sigma_t_M = ::expf(-sampling_altitude / m_Hm) * ray_step_size; // Mie varying term
					integration_partial_sigma_t_R += partial_sigma_t_R; // Integrate Rayleigh partial optical depth
					integration_partial_sigma_t_M += partial_sigma_t_M; // Integrate Mie partial optical depth

					// 3. Compute in-scattering from the sun (Li) for this sample point
					float light_ray_t_enter, light_ray_t_exit;
					Ray ray_Li = Ray(sample_position, m_sunDirection); // Ray from sample point to sun
					intersectSphere(ray_Li, make_float3(0.0f),
						m_atmosphereRadiusMeters, &light_ray_t_enter, &light_ray_t_exit);
					const float light_ray_step_size = light_ray_t_exit / lightRayMarchingSteps; // Step size for light ray raymarching

					float current_light_ray_t = 0.0f;
					float integration_partial_sigma_t_R_Li = 0.0f, integration_partial_sigma_t_M_Li = 0.0f; // Integrated partial optical depths for light ray

					// Raymarch along the light ray to the sun
					uint8_t j;
#pragma unroll // Hint to the compiler to unroll this loop for performance (CUDA context)
					for (j = 0; j < lightRayMarchingSteps; ++j)
					{
						const float3 sample_position_Li = ray_Li.getPointAt(current_light_ray_t + (light_ray_step_size * 0.5f));
						const float altitude_Li = length(sample_position_Li) - m_earthRadiusMeters;
						if (altitude_Li < 0.0f) { // If light ray sample is below ground, terminate
							break; // Sun light is occluded by Earth
						}
						integration_partial_sigma_t_R_Li += ::expf(-altitude_Li / m_Hr) * light_ray_step_size; // Integrate Rayleigh partial optical depth for light ray
						integration_partial_sigma_t_M_Li += ::expf(-altitude_Li / m_Hm) * light_ray_step_size; // Integrate Mie partial optical depth for light ray
						current_light_ray_t += light_ray_step_size;
					}
					;
					if (j == lightRayMarchingSteps) // Light raymarch completed without hitting Earth
					{
						// Sea-level scattering coefficients (precomputed for sea level for efficiency)
						const RGBSpectrum sigma_t_R_sea_level = m_sigma_s_R_sea_level + m_sigma_a_R; // Attenuation coefficient = scattering + absorption
						const RGBSpectrum sigma_t_M_sea_level = m_sigma_s_M_sea_level + m_sigma_a_M;

						// Calculate total optical depth (tau) along both view ray and light ray paths
						const RGBSpectrum tau = sigma_t_R_sea_level * (integration_partial_sigma_t_R + integration_partial_sigma_t_R_Li)
							+ sigma_t_M_sea_level * (integration_partial_sigma_t_M + integration_partial_sigma_t_M_Li);

						const RGBSpectrum Tr = expf(-tau); // Transmittance along both paths

						integration_R_Tr += Tr * partial_sigma_t_R; // Accumulate Rayleigh term
						integration_M_Tr += Tr * partial_sigma_t_M;   // Accumulate Mie term
					}
					current_ray_t += ray_step_size;
				}

				// 4. Compute final in-scattered radiance
				const float mu = dot(p_ray.getDirection(), m_sunDirection); // Cosine of angle between view ray and sun direction

				const RGBSpectrum L_inscatter =
					(m_sigma_s_R_sea_level * integration_R_Tr * rayleighPhaseFunction(mu) // Rayleigh scattering contribution
						+ m_sigma_s_M_sea_level * integration_M_Tr * miePhaseFunction(mu)); // Mie scattering contribution

				return L_inscatter * Values::SUN_HORIZON_LUMINANCE_NITS * m_sunEmissionFactor; // Scale by sun luminance and emission factor
			}

			/**
			 * @brief Gets the luminance of the atmosphere (scaled sun horizon luminance).
			 *
			 * @return Luminance value in nits.
			 */
			__device__ float getAtmosphereLuminance() const {
				return m_sunEmissionFactor * Values::SUN_HORIZON_LUMINANCE_NITS;
			}

			/**
			 * @brief Gets the Earth's radius in meters.
			 *
			 * @return Earth radius in meters.
			 */
			__device__ float getEarthRadiusMeters() const {
				return m_earthRadiusMeters;
			}

			/**
			 * @brief Gets the normalized sun direction vector.
			 *
			 * @return Normalized sun direction vector.
			 */
			__device__ float3 getSunDirection() const {
				return m_sunDirection;
			}

		private:
			/**
			 * @brief Rayleigh phase function.
			 *
			 * Implements the Rayleigh phase function, describing the angular distribution
			 * of scattering by particles much smaller than the wavelength of light.
			 * Formula derived from [Reference to Rayleigh scattering theory].
			 *
			 * @param cos_theta Cosine of the scattering angle (angle between incident and scattered directions).
			 * @return Phase function probability value.
			 */
			__device__ float rayleighPhaseFunction(float cos_theta) const
			{
				const float numerator = 3.0f * (1.0f + Sqr(cos_theta));
				const float probability = numerator / (16.0f * Constants::PI);
				return probability;
			}

			/**
			 * @brief Mie phase function (simplified, Henyey-Greenstein like approximation).
			 *
			 * Implements a simplified Mie phase function, approximating forward scattering
			 * behavior typical of Mie scattering by larger particles (aerosols).
			 * This is a variant of the Henyey-Greenstein phase function.
			 * Parameters are chosen to approximate atmospheric Mie scattering.
			 *
			 * @param cos_theta Cosine of the scattering angle.
			 * @return Phase function probability value.
			 */
			__device__ float miePhaseFunction(float cos_theta) const
			{
				constexpr float mieAnisotropyFactor = 0.76f; // g parameter - anisotropy factor, controls forward scattering
				const float denom = (8.0f * Constants::PI) * ((2.0f + Sqr(mieAnisotropyFactor)) * pow(1.0f + Sqr(mieAnisotropyFactor) - 2.0f * mieAnisotropyFactor * cos_theta, 1.5f)); // 3.0f/2.0f = 1.5f
				const float probability = 3.0f * ((1.0f - Sqr(mieAnisotropyFactor)) * (1.0f + Sqr(cos_theta))) / denom;
				return probability;
			}

			// --- Member Variables ---

			// Raymarching parameters - control quality and performance
			const uint32_t rayMarchingSteps{ 16u };          ///< Number of steps for primary ray raymarching
			const uint32_t lightRayMarchingSteps{ 8u };        ///< Number of steps for light ray raymarching (in-scattering)

			// Sun parameters
			float3 m_sunDirection = make_float3(0.0f, 1.0f, 0.0f); ///< Normalized direction to the sun
			const float m_sunEmissionFactor{ 1.0f };           ///< Emission scaling factor for sun intensity

			// Physical constants (IAU Commission 4 and standard atmospheric values)
			const float m_earthRadiusMeters{ 6'378.1370_km };      ///< Earth radius in meters (IAU Commission 4 value)
			const float m_atmosphereRadiusMeters{ 6'438.1370_km }; ///< Atmosphere radius in meters (Earth radius + 60km atmosphere extent)
			const float m_Hr{ 7'994.0_metres };                  ///< Rayleigh scale height in meters (atmospheric thickness for uniform density)
			const float m_Hm{ 1'200.0_metres };                  ///< Mie scale height in meters

			// Scattering coefficients at sea level (Beta values from Nishita paper, precomputed for sea-level)
			const RGBSpectrum m_sigma_s_R_sea_level = RGBSpectrum(3.8e-6f, 13.5e-6f, 33.1e-6f); ///< Rayleigh scattering coefficients (sea level)
			const RGBSpectrum m_sigma_s_M_sea_level = RGBSpectrum(21e-6f);                    ///< Mie scattering coefficients (sea level)

			const float m_sigma_a_R{ 0.0f };                       ///< Rayleigh absorption coefficient (set to 0.0f in this model - no Rayleigh absorption)
			const float m_sigma_a_M{ 21e-6f * 0.1f };
		};
	}/*Atmosphere*/
}/*KittlesPT*/