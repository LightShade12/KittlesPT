#pragma once
#include "vector_types_extension.cuh"

namespace KittlesPT
{
	inline __device__ __host__ bool operator!(const float3& vec)
	{
		return (vec.x == 0.f && vec.y == 0.f && vec.z == 0.f);
	}

	inline __device__ float AbsDot(const float3& a, const float3& b)
	{
		return fabsf(dot(a, b));
	}

	inline __device__ bool checkNaN(const float3& vec)
	{
		return isnan(vec.x) || isnan(vec.y) || isnan(vec.z);
	}

	inline __device__ bool checkINF(const float3& vec)
	{
		return isinf(vec.x) || isinf(vec.y) || isinf(vec.z);
	}

	inline __device__ float3 clampOutput(const float3& v)
	{
		if ((checkNaN(v)) || (checkINF(v)))
		{
			return make_float3(0);
		}
		return clamp(v, 0, 1.0e8f);
	}

	inline __device__ float3 log2f(const float3 a)
	{
		return make_float3(::log2f(a.x), ::log2f(a.y), ::log2f(a.z));
	}

	inline __device__ float3 powf(const float3 a, const float3 b)
	{
		return make_float3(::powf(a.x, b.x), ::powf(a.y, b.y), ::powf(a.z, b.z));
	}

	//TODO: is Z-up only for tangent space; fix inconsistency
	inline __device__ float3 sphericalToCartesian(float theta, float phi)
	{
		float3 wm;
		wm.x = sinf(theta) * cosf(phi);
		wm.y = sinf(theta) * sinf(phi);
		wm.z = cosf(theta);
		return wm;
	}

	__device__ bool refract(const float3& wi, float3 normal, float ior, float3& wt);

	inline __device__ bool sameHemisphere(const float3& a, const float3& b, const float3& n)
	{
		return (dot(a, n) * dot(b, n)) > 0.0f;
	}
}