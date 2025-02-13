#pragma once
#include "matrix_maths.cuh" //brings all the rest of headers
#include "constants.cuh"

namespace KittlesPT
{
	__host__ __device__ inline float radians(float degree)
	{
		return degree * Constants::RADS_PER_DEG;
	}

	__host__ __device__ inline float degrees(float radians)
	{
		return radians * Constants::DEGS_PER_RAD;
	}

	__host__ __device__ inline float inversesqrt(float v)
	{
		return rsqrtf(v);
	}

	__host__ __device__ inline float step(float edge, float v)
	{
		return (v > edge) ? 1.0f : 0.0f;
	}

	__host__ __device__ inline float Sqr(float v)
	{
		return v * v;
	}

	__host__ __device__ inline float sign(float v) {
		return (v < 0.0f) ? -1.0f : (v > 0.0f) ? 1.0f : 0.0f;
	}
	__host__ __device__ inline float sign(int v) {
		return (v < 0) ? -1.0f : (v > 0) ? 1.0f : 0.0f;
	}

	__host__ __device__ inline uint saturate(uint v) {
		return clamp(v, 0u, 1u);
	}
	__host__ __device__ inline uint2 saturate(uint2 v) {
		return clamp(v, 0, 1);
	}
	__host__ __device__ inline uint3 saturate(uint3 v) {
		return clamp(v, 0, 1);
	}
	__host__ __device__ inline uint4 saturate(uint4 v) {
		return clamp(v, 0, 1);
	}

	__host__ __device__ inline int saturate(int v) {
		return clamp(v, 0, 1);
	}
	__host__ __device__ inline int2 saturate(int2 v) {
		return clamp(v, 0, 1);
	}
	__host__ __device__ inline int3 saturate(int3 v) {
		return clamp(v, 0, 1);
	}
	__host__ __device__ inline int4 saturate(int4 v) {
		return clamp(v, 0, 1);
	}

	__host__ __device__ inline float saturate(float v) {
		return clamp(v, 0.0f, 1.0f);
	}
	__host__ __device__ inline float2 saturate(float2 v) {
		return clamp(v, 0.0f, 1.0f);
	}
	__host__ __device__ inline float3 saturate(float3 v) {
		return clamp(v, 0.0f, 1.0f);
	}
	__host__ __device__ inline float4 saturate(float4 v) {
		return clamp(v, 0.0f, 1.0f);
	}

	__host__ __device__ inline float distance(float2 p0, float2 p1) {
		return length(p0 - p1);
	}
	__host__ __device__ inline float distance(float3 p0, float3 p1) {
		return length(p0 - p1);
	}
	__host__ __device__ inline float distance(float4 p0, float4 p1) {
		return length(p0 - p1);
	}

	/*
	__device__ float Gaussian(float x, float mu, float sigma)
	{
		return (1.0f / sqrtf(2.0f * Constants::PI * sigma * sigma) *
			expf(-Sqr(x - mu) / (2.0f * sigma * sigma)));
	}

	__constant__ constexpr float ONE_MINUS_EPSILON = 0x1.fffffep-1;

	__device__ float sampleLinear(float u, float a, float b)
	{
		if (u == 0 && a == 0) {
			return 0;
		}
		float x = u * (a + b) / (a + sqrtf(lerp(Sqr(a), Sqr(b), u)));
		return fminf(x, ONE_MINUS_EPSILON);
	}

	__device__ float nextFloatDown(float v)
	{
		// Handle infinity and positive zero for _NextFloatDown()_
		if (isinf(v) && v < 0.) {
			return v;
		}
		if (v == 0.f) {
			v = -0.f;
		}
		uint32_t ui = floatBitsToUint(v);
		if (v > 0) {
			--ui;
		}
		else {
			++ui;
		}
		return uintBitsToFloat(ui);
	}

	__device__ int sampleDiscrete(const float* weights, int w_size, float u, float* uRemapped)
	{
		if (w_size == 0)
		{
			return -1;
		}

		float sumWeights = 0;
		for (int i = 0; i < w_size; i++) {
			sumWeights += weights[i];
		}

		float up = u * sumWeights;
		if (up == sumWeights) {
			up = nextFloatDown(up);
		}

		int offset = 0;
		float sum = 0;

		while (sum + weights[offset] <= up) {
			sum += weights[offset++];
		}

		if (uRemapped) {
			*uRemapped = fminf((up - sum) / weights[offset], ONE_MINUS_EPSILON);
		}

		return offset;
	}

	__device__ float sampleTent(float u, float r)
	{
		constexpr float weights[2] = { 0.5f, 0.5f };
		if (sampleDiscrete(weights, 2, u, &u) == 0) {
			return -r + r * sampleLinear(u, 0, 1);
		}
		else {
			return r * sampleLinear(u, 1, 0);
		}
	}
	*/
}