#pragma once
#include "matrix_maths.cuh" //brings all the rest of headers
#include "constants.cuh"

namespace KittlesPT
{
	__host__ __device__ inline float deg2rad(float degree)
	{
		return degree * Constants::RADS_PER_DEG;
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

	__device__ float Gaussian(float x, float mu = 0, float sigma = 1);

	__device__ float sampleLinear(float u, float a, float b);

	__device__ float nextFloatDown(float v);

	__device__ int sampleDiscrete(const float* weights, int w_size, float u, float* uRemapped);

	__device__ float sampleTent(float u, float r);
}