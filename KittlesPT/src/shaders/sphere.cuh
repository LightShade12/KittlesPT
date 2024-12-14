#pragma once

#include "ray.cuh"
#include "../maths/linear_algebra.cuh"

namespace KittlesPT
{
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
		float3 world_normal;
	};

	class Sphere
	{
	public:
		__device__ __host__ Sphere(float radius_, float3 pos)//TODO: make it host only
			:radius(radius_), world_position(pos) {};

		__device__ Intersection intersect(const Ray& ray) const;

		float3 world_position;
		float radius = 1;
	};
}