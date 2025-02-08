#pragma once
#include "ray.cuh"
#include "bsdf.cuh"
#include <vector_types.h>
#include <cstdint>

namespace KittlesPT
{
	struct ShaderData;
	class RGBSpectrum;
	class Ray;
	class BSDF;
	class Light;

	struct SurfaceInteraction
	{
		__device__ RGBSpectrum Le(const ShaderData& shader_data, const Ray& ray) const;

		__device__ BSDF getBSDF(const ShaderData& shader_data) const;

		__device__ void skipInteraction(Ray* ray)
		{
			float3 offset = world_geometric_normal * Constants::HIT_EPSILON;
			if (dot(ray->getDirection(), world_geometric_normal) < 0) {
				offset *= -1.0f;
			}
			float3 orig = world_position + offset;
			*ray = Ray(orig, ray->getDirection());
		}

		__device__ Ray spawnRay(float3 wi, int scatter_flags) const
		{
			float3 ray_orig;
			if (scatter_flags & BSDFSample::Transmitted) {
				ray_orig = world_position - (world_geometric_normal * Constants::HIT_EPSILON);
			}
			else {
				ray_orig = world_position + (world_geometric_normal * Constants::HIT_EPSILON);
			}
			return Ray(ray_orig, wi);
		}

		__device__ Ray spawnRayTo(float3 target) const
		{
			float3 ray_orig;
			if (backface) {
				ray_orig = world_position - (world_geometric_normal * Constants::HIT_EPSILON);
			}
			else {
				ray_orig = world_position + (world_geometric_normal * Constants::HIT_EPSILON);
			}
			return Ray(ray_orig, normalize(target - ray_orig));
		}

		//--------------------------------------------------
		float3 wo{ 0.0f,0.0f,0.0f };
		const Light* arealight = nullptr;
		int32_t material_id = -1;
		float distance = INFINITY;
		float2 uv{ 0.0f,0.0f };
		float3 world_position{ 0.0f,0.0f,0.0f };
		float3 world_geometric_normal{ 0.0f,0.0f,0.0f };
		bool backface = false;
	};

	struct Intersection
	{
		//closest hit function
		__device__ SurfaceInteraction getSurfaceInteraction(const ShaderData& shader_data, const Ray& ray);

		__device__ bool operator!()
		{
			return (primitive_id < 0);
		}
		__device__ operator bool()
		{
			return (primitive_id >= 0);
		}

		//-------------------------------------
		float distance = INFINITY;
		int32_t primitive_id = -1;
		int32_t instance_id = -1;
		float3 bary_coords{ 0.0f,0.0f,0.0f };
	};
}/*KittlesPT*/