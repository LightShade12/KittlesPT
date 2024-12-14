#pragma once
#include "../maths/vector_maths.cuh"

namespace KittlesPT
{
	class Ray
	{
	public:
		__device__ Ray(float3 orig, float3 dir) :
			origin(orig), direction(dir), inv_direction(1 / dir) {};
		__device__ float3 getOrigin() const { return origin; }
		__device__ float3 getDirection() const { return direction; }
		__device__ float3 getInvDirection() const { return inv_direction; }
		__device__ float3 getPointAt(float dist) const { return origin + (dist * direction); };
	private:
		float3 origin;
		float3 direction;
		float3 inv_direction;
	};
}