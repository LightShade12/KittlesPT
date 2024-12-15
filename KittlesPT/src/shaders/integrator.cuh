#pragma once
#include "containers.cuh"
#include "sphere.cuh"
#include "bsdf.cuh"
#include "samplers.cuh"
#include "atmosphere.cuh"

namespace KittlesPT
{
	namespace Integrator
	{
		__device__ SurfaceInteraction closestHit(const Ray& ray, const Intersection& intr, const Sphere& sp)
		{
			SurfaceInteraction surfintr;

			surfintr.world_position = ray.getPointAt(intr.distance);
			surfintr.distance = intr.distance;
			surfintr.world_normal = normalize(surfintr.world_position - sp.world_position);

			return surfintr;
		}

		__device__ Intersection intersect(const GlobalShaderData& shader_data, const Ray& ray)
		{
			Intersection closest;
			closest.distance = INFINITY;

			for (int instance_id = 0; instance_id < shader_data.scene_buffer.num; instance_id++)
			{
				const Sphere& sphere = shader_data.scene_buffer.data[instance_id];
				Intersection intr = sphere.intersect(ray);
				if (intr.distance < closest.distance && intr.distance >= 0)
				{
					closest.distance = intr.distance;
					closest.instance_id = instance_id;
				}
			}
			return closest;
		}

		__device__ float3 sensorL(const GlobalShaderData& shader_data, const Ray& ray_in, IndependentSampler& sampler)
		{
			float3 light = make_float3(0);
			float3 throughput = make_float3(1);
			Atmosphere atmosphere(normalize(make_float3(1, 1, 1)), 20.0f);

			constexpr int MAX_RAY_DEPTH = 3;
			Ray ray = ray_in;

			for (int bounce_depth = 0; bounce_depth < MAX_RAY_DEPTH; bounce_depth++)
			{
				sampler.setSeed(sampler.getSeed() + bounce_depth);

				Intersection intr = intersect(shader_data, ray);

				if (!intr)
				{
					//miss
					float3 unit_direction = normalize(ray.getDirection());
					float a = 0.5 * (unit_direction.y + 1.0);

					//float3 color = (1.0 - a) * make_float3(1.0, 1.0, 1.0) + a * make_float3(0.5, 0.7, 1.0);
					float3 color = atmosphere.Le(make_float3(0, atmosphere.m_earthRadius + 1, 0),
						normalize(ray.getDirection()), 0, FLT_MAX);
					light += color * throughput;
					break;
				}

				//hit
				float3 wo = -ray.getDirection();

				SurfaceInteraction surfintr = closestHit(ray, intr, shader_data.scene_buffer.data[intr.instance_id]);

				BSDF bsdf = BSDF(generateOrthonormalBasis(surfintr.world_normal), make_float3(0.8, 0, 0), 0.1);
				BSDFSample bs = bsdf.sampleBSDF(wo, sampler.get2D(), sampler.get2D());

				float3 wi = bs.wi;
				float pdf = bs.pdf;

				float3 fcos = bs.f * dot(surfintr.world_normal, wi);
				if (!fcos) break;

				throughput *= (fcos / pdf);

				ray = Ray(surfintr.world_position + (surfintr.world_normal * Constants::HIT_EPSILON), wi);
			}

			return light;
		}
	}
}