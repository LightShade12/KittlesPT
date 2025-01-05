#include "integrator.cuh"
#include "containers.cuh"
#include "interaction.cuh"
#include "sphere.cuh"
#include "bsdf.cuh"
#include "samplers.cuh"
#include "atmosphere.cuh"
#include "light_sampler.cuh"
#include "ray.cuh"
#include "maths/sampling.cuh"
#include "color.cuh"
#include "packing.cuh"

namespace KittlesPT
{
	namespace Integrator
	{
		__device__ Intersection intersect(const GlobalShaderData& shader_data, const Ray& ray, float tmax)
		{
			Intersection closest;
			closest.distance = INFINITY;
			Intersection intr;

			for (int instance_id = 0; instance_id < shader_data.geometry_buffer.num; instance_id++)
			{
				const Sphere& sphere = shader_data.geometry_buffer.data[instance_id];
				intr = sphere.intersect(ray, tmax);//currently only returns a float; triangle will return struct
				if (intr.distance < closest.distance && intr.distance >= 0 && intr.distance < tmax)
				{
					closest.distance = intr.distance;
					closest.instance_id = instance_id;
				}
			}
			return closest;
		}

		__device__ bool intersectShadow(const GlobalShaderData& shader_data, const Ray& ray, float tmax)
		{
			Intersection intr;
			for (int instance_id = 0; instance_id < shader_data.geometry_buffer.num; instance_id++)
			{
				const Sphere& sphere = shader_data.geometry_buffer.data[instance_id];
				intr = sphere.intersect(ray, tmax);
				if (intr.distance >= 0 && intr.distance < tmax)
				{
					return true;
				}
			}
			return false;
		}

		__device__ bool Unoccluded(const GlobalShaderData& shader_data, const SurfaceInteraction& surface, float3 target)
		{
			constexpr float SHADOWRAY_EPSILON = 0.11f;//TODO: put this in a constants file
			Ray shadow_ray = surface.spawnRayTo(target);
			float tmax = length(target - shadow_ray.getOrigin()) - SHADOWRAY_EPSILON;
			return (!intersectShadow(shader_data, shadow_ray, tmax));
		}

		__device__ RGBSpectrum sampleLdSun(const GlobalShaderData& shader_data, const Ray& ray, float3 sun_direction, const BSDF& bsdf,
			const SurfaceInteraction& surface, const Atmosphere& atmosphere, IndependentSampler& sampler)
		{
			RGBSpectrum Ld(0);
			float3 sun_position = sun_direction * SUN_DISTANCE_METERS;
			float sun_radius = angularDiameterToPhysicalDiameter(
				shader_data.procedural_environment_data.sun_angular_diameter_rad,
				SUN_DISTANCE_METERS) / 2.0f;
			float3 sample_offset = make_float3(sampler.get2D() * 2 - 1, sampler.get1D() * 2 - 1);
			float3 target = sun_position + (sample_offset * sun_radius);

			float3 wo = -ray.getDirection();
			float3 atmosphere_observer_position = make_float3(0, atmosphere.m_earth_radius + 1, 0);

			//TODO: fix cosine for refractive caustics
			RGBSpectrum fcos = bsdf.f(wo, sun_direction) *
				fmaxf(0, dot(sun_direction, ((surface.backface) ? -1.0f : 1.0f) * surface.world_geometric_normal));

			if (!fcos)
			{
				return Ld;
			}

			RGBSpectrum sun_color = atmosphere.Le(atmosphere_observer_position, sun_direction, 0, INFINITY);

			if (!sun_color)
			{
				return Ld;
			}

			if (!Unoccluded(shader_data, surface, target))
			{
				return Ld;
			};

			float sun_area = Constants::PI * Sqr(sun_radius);
			float3 sun_n = normalize(target - sun_position), wi = normalize(target - surface.world_position);
			float cos_sun = AbsDot(sun_n, -wi);
			float pdf = (1.0f / sun_area) / (cos_sun / Sqr(SUN_DISTANCE_METERS));
			//TODO: pbr values; better sun sampling/pdf
			Ld = (fcos * sun_color * 5000.0f * shader_data.procedural_environment_data.sun_radiance_intensity) / pdf;

			return Ld;
		}

		__device__ RGBSpectrum sampleLd(const GlobalShaderData& shader_data, const Ray& ray, const BSDF& bsdf,
			const SurfaceInteraction& surface, const LightSampler& light_sampler, IndependentSampler& sampler)
		{
			RGBSpectrum Ld(0);

			SampledLight sampled_light = light_sampler.sample(sampler.get1D());

			//empty buffer
			if (!sampled_light)
			{
				return Ld;
			}

			LightLiSample ls = sampled_light.light->sampleLi(shader_data, LightSampleContext(surface), sampler.get2D());
			if (!ls)
			{
				return Ld;
			}

			float3 wi = ls.wi;
			float3 wo = -ray.getDirection();
			RGBSpectrum fcos = bsdf.f(wo, wi) *
				fmaxf(0, dot(wi, ((surface.backface) ? -1.0f : 1.0f) * surface.world_geometric_normal));;

			if (!fcos)
			{
				return Ld;
			}

			if (!Unoccluded(shader_data, surface, ls.wpos_light))
			{
				return Ld;
			}

			float p_l = (sampled_light.probability * ls.pdf);
			float p_b = bsdf.pdf(wo, wi);
			float w_l = powerHeuristic(1, p_l, 1, p_b);

			Ld = (ls.L * fcos * w_l) / p_l;

			return Ld;
		}

		//russian roulette
		__device__ bool russianRoulette(RGBSpectrum& throughput, float eta_scale,
			int bounce_depth, IndependentSampler& sampler)
		{
			RGBSpectrum rr_beta = throughput * eta_scale;
			if (rr_beta.maxComponentValue() < 1 && bounce_depth > 1) {
				float q = fmaxf(0.0f, 1.0f - rr_beta.maxComponentValue());
				if (sampler.get1D() < q)
				{
					return true;
				}
				throughput /= (1.0f - q);
			}
			return false;
		}

		__device__ float3 sphericalToSunDirection(float theta, float phi)
		{
			return normalize(make_float3(
				cosf(phi) * cosf(theta),
				sinf(theta),
				sinf(phi) * cosf(theta)
			));
		}

		__device__ RGBSpectrum Li(const GlobalShaderData& shader_data, const Ray& ray_in,
			IndependentSampler& sampler, GBuffer* visible_surface)
		{
			RGBSpectrum light(0.0f);
			RGBSpectrum throughput(1.0f);

			float3 sun_direction = sphericalToSunDirection(
				shader_data.procedural_environment_data.sun_theta_rad,
				shader_data.procedural_environment_data.sun_phi_rad);

			Atmosphere atmosphere(sun_direction, shader_data.procedural_environment_data.sun_radiance_intensity);
			float3 atmosphere_observer_position = make_float3(0, atmosphere.m_earth_radius + 1, 0);
			LightSampler light_sampler(shader_data.lights_buffer.data, shader_data.lights_buffer.num);

			float eta_scale = 1.0;
			float p_b = 1.0f;
			LightSampleContext prev_ctx;

			const int MAX_RAY_DEPTH = shader_data.pathtracer_settings.max_bounce_depth;
			Ray ray = ray_in;

			for (int bounce_depth = 0; bounce_depth <= MAX_RAY_DEPTH; bounce_depth++)
			{
				sampler.setSeed(sampler.getSeed() + bounce_depth);
				bool first_surface = (bounce_depth == 0);

				Intersection intr = intersect(shader_data, ray, INFINITY);

				if (!intr)
				{
					//miss
					RGBSpectrum sky_radiance = atmosphere.Le(atmosphere_observer_position,
						ray.getDirection(), 0, INFINITY);
					light += sky_radiance * throughput;
					break;
				}
				//hit

				SurfaceInteraction surfintr = intr.getSurfaceInteraction(shader_data, ray);
				float3 wo = -ray.getDirection();

				//Le
				{
					const Light* arealight = surfintr.light;
					float w_l = 1.0f;
					if (arealight && !first_surface)
					{
						float light_pdf = light_sampler.PMF(arealight) * arealight->pdf_Li(prev_ctx, LightLiSample(surfintr));
						w_l = powerHeuristic(1, p_b, 1, light_pdf);
					}
					light += surfintr.Le(shader_data, ray) * w_l * throughput;
				}

				BSDF bsdf = surfintr.getBSDF(shader_data);

				if (first_surface)
				{
					*visible_surface = GBuffer(bsdf.albedo_factor, surfintr);
				}

				RGBSpectrum Ld = sampleLd(shader_data, ray,
					bsdf, surfintr, light_sampler, sampler);
				light += Ld * throughput;

				RGBSpectrum Ld_sun = sampleLdSun(shader_data, ray, sun_direction,
					bsdf, surfintr, atmosphere, sampler);
				light += Ld_sun * throughput;

				BSDFSample bs = bsdf.sampleBSDF(wo, sampler.get2D(), sampler.get2D());
				if (bs.scatterTypeIs(BSDFSample::Absorbed))
				{
					break;
				}

				const float3& wi = bs.wi;
				float pdf = bs.pdf;

				//uses absdot for allowing refraction
				RGBSpectrum fcos = bs.f * AbsDot(surfintr.world_geometric_normal, wi);
				if (!fcos)
				{
					break;
				}

				throughput *= (fcos / pdf);

				p_b = bsdf.pdf(wo, wi);
				prev_ctx = LightSampleContext(surfintr);
				ray = surfintr.spawnRay(wi, bs.scatter);

				//russian roulette
				RGBSpectrum rr_beta = throughput * eta_scale;
				if (rr_beta.maxComponentValue() < 1 && bounce_depth > 1) {
					float q = fmaxf(0.0f, 1.0f - rr_beta.maxComponentValue());
					if (sampler.get1D() < q)
					{
						break;
					}
					throughput /= (1.0f - q);
				}
			}

			return light;
		}
	}
}