#pragma once
#include "containers.cuh"
#include "sphere.cuh"
#include "bsdf.cuh"
#include "samplers.cuh"
#include "material.cuh"
#include "atmosphere.cuh"

namespace KittlesPT
{
	namespace Integrator
	{
		__device__ SurfaceInteraction closestHit(const Ray& ray, const Intersection& intr, const Sphere& sp)
		{
			SurfaceInteraction surfintr;
			float3 wo = -ray.getDirection();
			surfintr.world_position = ray.getPointAt(intr.distance);
			surfintr.distance = intr.distance;
			surfintr.world_geometric_normal = normalize(surfintr.world_position - sp.world_position);
			if (dot(surfintr.world_geometric_normal, wo) < 0)
			{
				surfintr.world_geometric_normal *= -1.0f;
				surfintr.backface = true;
			}
			surfintr.material_id = sp.material_id;

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

		__device__ float3 sensorRadiance(const GlobalShaderData& shader_data, const Ray& ray_in, IndependentSampler& sampler)
		{
			float3 light = make_float3(0);
			float3 throughput = make_float3(1);
			Atmosphere atmosphere(normalize(make_float3(-1, 1, -1)), 20.0f);

			constexpr int MAX_RAY_DEPTH = 5;//TODO: put in a constants file or sumn?
			Ray ray = ray_in;

			for (int bounce_depth = 0; bounce_depth < MAX_RAY_DEPTH; bounce_depth++)
			{
				sampler.setSeed(sampler.getSeed() + bounce_depth);

				Intersection intr = intersect(shader_data, ray);

				if (!intr)
				{
					//miss

					float3 color = atmosphere.Le(make_float3(0, atmosphere.m_earthRadius + 1, 0),
						normalize(ray.getDirection()), 0, FLT_MAX);
					light += color * throughput;
					break;
				}

				//hit
				float3 wo = -ray.getDirection();

				SurfaceInteraction surfintr = closestHit(ray, intr, shader_data.scene_buffer.data[intr.instance_id]);
				BSDF bsdf = surfintr.getBSDF(shader_data);
				BSDFSample bs = bsdf.sampleBSDF(wo, sampler.get2D(), sampler.get2D());

				if (bs.scatterTypeIs(BSDFSample::Absorbed)) { break; }

				float3 wi = bs.wi;
				float pdf = bs.pdf;

				float3 fcos = bs.f * AbsDot(surfintr.world_geometric_normal, wi);
				if (!fcos) { break; }

				throughput *= (fcos / pdf);

				ray = surfintr.spawnRay(wi, bs.scatter);
			}

			return light;
		}
	}
}