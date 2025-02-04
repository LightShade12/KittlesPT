#pragma once
#include "shaders/ray.cuh"

namespace KittlesPT {
	class Bounds3f {
	public:

		Bounds3f() = default;
		__host__ Bounds3f(float3 min, float3 max) :
			pmin(min), pmax(max) {};

		__host__ __device__ float3 centroid() const
		{
			return 0.5f * pmin + 0.5f * pmax;
		}

		__host__ __device__ float3 diagonal() const {
			return pmax - pmin;
		}

		__host__ __device__ float surfaceArea() const
		{
			float3 d = diagonal();
			return 2.0f * (d.x * d.y + d.y * d.z + d.z * d.x);
		}

		__host__ void grow(float3 p)
		{
			pmin = fminf(pmin, p), pmax = fmaxf(pmax, p);
		}
		__host__ void grow(Bounds3f b)
		{
			grow(b.pmax);
			grow(b.pmin);
		}

		__device__ bool intersectP(const Ray& ray, float t_tmin, float t_tmax, float* hit0, float* hit1) const
		{
			float3 t0 = (pmin - ray.getOrigin()) * ray.getInvDirection();
			float3 t1 = (pmax - ray.getOrigin()) * ray.getInvDirection();

			float3 tmin = fminf(t0, t1);
			float3 tmax = fmaxf(t0, t1);

			float tenter = fmaxf(fmaxf(tmin.x, tmin.y), tmin.z);
			float texit = fminf(fminf(tmax.x, tmax.y), tmax.z);

			bool hit = (tenter <= texit) && (texit >= t_tmin) && (tenter <= t_tmax);

			if (hit) {
				*hit0 = tenter;
				*hit1 = texit;
			}

			return hit;
		}

	public:
		float3 pmin = constexpr_float3(FLT_MAX), pmax = constexpr_float3(-FLT_MAX);
	};
}