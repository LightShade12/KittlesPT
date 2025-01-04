#pragma once
#include "matrix_maths.cuh" //brings all the rest of headers

namespace KittlesPT
{
	__host__ __device__	float deg2rad(float degree);

	__host__ __device__ float Sqr(float v);

	__device__ float Gaussian(float x, float mu = 0, float sigma = 1);

	__device__ float sampleLinear(float u, float a, float b);

	__device__ float nextFloatDown(float v);

	__device__ int sampleDiscrete(const float* weights, int w_size, float u, float* pmf, float* uRemapped);

	__device__ float sampleTent(float u, float r);
}