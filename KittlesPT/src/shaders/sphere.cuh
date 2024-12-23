#pragma once

#include "ray.cuh"
#include "bsdf.cuh"
#include "maths/linear_algebra.cuh"

namespace KittlesPT
{
	struct GlobalShaderData;

	struct Intersection
	{
		__device__ bool operator ! ();
		float distance = -1;
		int instance_id = -1;
	};

	struct SurfaceInteraction
	{
		float distance = -1;
		float3 world_position;
		float3 world_geometric_normal;
		int material_id = -1;
		bool backface = false;

		__device__ BSDF getBSDF(const GlobalShaderData& shader_data);

		__device__ Ray spawnRay(float3 wi, int scatter_flags);
	};

	class Sphere
	{
	public:
		__device__ __host__ Sphere(float radius_, float3 pos, int material_id)//TODO: make it host only
			:radius(radius_), world_position(pos), material_id(material_id) {};

		__device__ Intersection intersect(const Ray& ray) const;

		int material_id = -1;
		float3 world_position;
		float radius = 1;
	};
}