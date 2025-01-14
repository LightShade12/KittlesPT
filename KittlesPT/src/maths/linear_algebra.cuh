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

	__device__ float Gaussian(float x, float mu = 0, float sigma = 1);

	__device__ float sampleLinear(float u, float a, float b);

	__device__ float nextFloatDown(float v);

	__device__ int sampleDiscrete(const float* weights, int w_size, float u, float* uRemapped);

	__device__ float sampleTent(float u, float r);
}