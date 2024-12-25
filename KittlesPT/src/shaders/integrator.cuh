#pragma once
#include "containers.cuh"
#include "sphere.cuh"
#include "bsdf.cuh"
#include "samplers.cuh"
#include "material.cuh"
#include "atmosphere.cuh"
#include "light_sampler.cuh"
#include "maths/sampling.cuh"
#include "color.cuh"

namespace KittlesPT
{
	namespace Integrator
	{
		__device__ Intersection intersect(const GlobalShaderData& shader_data, const Ray& ray, float tmax)
		{
			Intersection closest;
			closest.distance = INFINITY;

			for (int instance_id = 0; instance_id < shader_data.geometry_buffer.num; instance_id++)
			{
				const Sphere& sphere = shader_data.geometry_buffer.data[instance_id];
				Intersection intr = sphere.intersect(ray, tmax);
				if (intr.distance < closest.distance && intr.distance >= 0 && intr.distance < tmax)
				{
					closest.distance = intr.distance;
					closest.instance_id = instance_id;
				}
			}
			return closest;
		}

		__device__ bool intersectShadow(const GlobalShaderData& shader_data, const Ray& ray, float tmax, int light_id)
		{
			Intersection intr;
			for (int instance_id = 0; instance_id < shader_data.geometry_buffer.num; instance_id++)
			{
				if (instance_id == light_id)
				{
					continue;
				}

				const Sphere& sphere = shader_data.geometry_buffer.data[instance_id];
				intr = sphere.intersect(ray, tmax);
				if (intr.distance >= 0 && intr.distance < tmax)
				{
					return true;
				}
			}
			return false;
		}

		__device__ bool Unoccluded(const GlobalShaderData& shader_data, const SurfaceInteraction& surface, float3 target, int light_id)
		{
			//return true;
			constexpr float SHADOWRAY_EPSILON = 0.11f;//TODO: put this in a constants file
			Ray ray = surface.spawnRayTo(target);
			float tmax = length(target - ray.getOrigin()) - SHADOWRAY_EPSILON;
			return !(intersectShadow(shader_data, ray, tmax, light_id));
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

			LightLiSample ls = sampled_light.light->sampleLi(shader_data, surface, sampler.get2D());
			if (!ls)
			{
				return Ld;
			}

			float3 wi = ls.wi;
			float3 wo = -ray.getDirection();
			RGBSpectrum fcos = bsdf.f(wo, wi) * fmaxf(dot(wi, surface.world_geometric_normal), 0.0f);
			//fcos = RGBSpectrum(fmaxf(dot(wi, surface.world_geometric_normal), 0.0f));

			if (!fcos || !Unoccluded(shader_data, surface, ls.wpos_light, sampled_light.light->prim_id))
			{
				return Ld;
			}

			float dist = length(surface.world_position - ls.wpos_light);
			float cos_theta_emitter = AbsDot(wi, ls.geo_wnorm);
			float p_l = (sampled_light.probability * ls.pdf) * (1 / cos_theta_emitter) * Sqr(dist);
			p_l = Sqr(dist) * sampled_light.probability * ls.pdf;

			//float p_b = bsdf.pdf(wo, wi);
			//float w_l = powerHeuristic(1, p_l, 1, p_b);
			//Ld = fcos * w_l * ls.L / p_l;

			Ld = (ls.L * fcos) / (2.0f * p_l);
			//Ld = (ls.L / 2.0f) / p_l;

			return Ld;
		}

		__device__ RGBSpectrum sensorRadiance(const GlobalShaderData& shader_data, const Ray& ray_in, IndependentSampler& sampler)
		{
			RGBSpectrum light(0.0f);
			RGBSpectrum throughput(1.0f);

			Atmosphere atmosphere(normalize(make_float3(-1, 1, -1)), 50.0f);
			LightSampler light_sampler(shader_data.lights_buffer.data, shader_data.lights_buffer.num);

			constexpr int MAX_RAY_DEPTH = 5;//TODO: put in a constants file or sumn?
			Ray ray = ray_in;

			for (int bounce_depth = 0; bounce_depth < MAX_RAY_DEPTH; bounce_depth++)
			{
				sampler.setSeed(sampler.getSeed() + bounce_depth);

				Intersection intr = intersect(shader_data, ray, INFINITY);

				if (!intr)
				{
					//miss
					float3 atmosphere_observer_position = make_float3(0, atmosphere.m_earth_radius + 1, 0);
					RGBSpectrum sky_radiance = atmosphere.Le(atmosphere_observer_position,
						normalize(ray.getDirection()), 0, FLT_MAX);
					light += sky_radiance * throughput;
					break;
				}

				//hit
				float3 wo = -ray.getDirection();

				SurfaceInteraction surfintr = intr.getSurfaceInteraction(shader_data, ray);

				if (bounce_depth == 0 && false)
				{
					light += surfintr.Le(shader_data, ray) * throughput;
				}

				BSDF bsdf = surfintr.getBSDF(shader_data);

				//light += sampleLd(shader_data, ray,
				//	bsdf, surfintr, light_sampler, sampler) * throughput;

				BSDFSample bs = bsdf.sampleBSDF(wo, sampler.get2D(), sampler.get2D());

				if (bs.scatterTypeIs(BSDFSample::Absorbed)) { break; }

				const float3& wi = bs.wi;
				float pdf = bs.pdf;

				RGBSpectrum fcos = bs.f * AbsDot(surfintr.world_geometric_normal, wi);
				if (!fcos) { break; }

				throughput *= (fcos / pdf);

				ray = surfintr.spawnRay(wi, bs.scatter);
			}

			return light;
		}
	}
}