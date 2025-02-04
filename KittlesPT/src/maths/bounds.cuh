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

		/*
		__host__ __device__ void adaptBounds(const Mat4& model_mat, const Bounds3f& original)
		{
			// Original Bounds3f corners before transformation
			float3 corners[8] = {
				original.pmin,                                 // (pmin.x, pmin.y, pmin.z)
				make_float3(original.pmax.x, original.pmin.y, original.pmin.z),  // (pmax.x, pmin.y, pmin.z)
				make_float3(original.pmin.x, original.pmax.y, original.pmin.z),  // (pmin.x, pmax.y, pmin.z)
				make_float3(original.pmin.x, original.pmin.y, original.pmax.z),  // (pmin.x, pmin.y, pmax.z)
				make_float3(original.pmax.x, original.pmax.y, original.pmin.z),  // (pmax.x, pmax.y, pmin.z)
				make_float3(original.pmin.x, original.pmax.y, original.pmax.z),  // (pmin.x, pmax.y, pmax.z)
				make_float3(original.pmax.x, original.pmin.y, original.pmax.z),  // (pmax.x, pmin.y, pmax.z)
				original.pmax                                  // (pmax.x, pmax.y, pmax.z)
			};

			// Variables to hold new pmin and pmax points
			float3 newMin = make_float3(std::numeric_limits<float>::max());
			float3 newMax = make_float3(-std::numeric_limits<float>::max());

			// Transform all 8 corners and compute new bounds
			for (int i = 0; i < 8; ++i) {
				// Transform the corner by the matrix (assume `transform` is a 4x4 matrix)
				float3 transformedPoint = make_float3(model_mat * make_float4(corners[i], 1.0f));

				// Update the new bounding box
				newMin = fminf(newMin, transformedPoint); // fminf compares each component (x, y, z)
				newMax = fmaxf(newMax, transformedPoint); // fmaxf compares each component (x, y, z)
			}

			// Set the new bounds
			pmin = newMin;
			pmax = newMax;
		}
		*/

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