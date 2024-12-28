#pragma once
#include <cuda_runtime.h>
#include <cstdint>

namespace KittlesPT
{
	class RNG
	{
	public:
		__device__ uint32_t pcg_hash(uint32_t input);
	};

	class IndependentSampler
	{
	public:

		__device__ void initPixelSeed(int2 pixel, int width, int index, int dimension = 0);

		__device__ float get1D();

		__device__ uint32_t getSeed() { return seed; }
		__device__ void setSeed(uint32_t x) { seed = x; }

		__device__ float2 get2D();

	public:
		uint32_t seed;
		RNG rng;
	};

	//TODO: learn more about these
	__device__ float2 sampleUniformDiskPolar(float2 u);

	//NOTE: All this sampling is done in tangent space
	__device__ float3 sampleUniformHemisphere(float2 u);

	__device__ float3 sampleUniformSphere(float2 xi);

	__device__ float3 sampleCosineWeightedHemisphere(float2 xi);

	__device__ float3 toSphericalDirection(float sin_theta, float cos_theta, float phi);

	__device__ float3 toSphericalDirection(float theta, float phi);

	__device__ float3 sampleUniformCone(float2 u, float cos_theta_max);
}