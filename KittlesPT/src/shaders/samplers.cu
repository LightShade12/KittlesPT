#include "samplers.cuh"
#include "maths/constants.cuh"
#include "maths/vector_maths.cuh"

namespace KittlesPT
{
	__device__ uint32_t RNG::pcg_hash(uint32_t input)
	{
		uint32_t state = input * 747796405u + 2891336453u;
		uint32_t word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
		return (word >> 22u) ^ word;
	}
	__device__ void IndependentSampler::initPixelSeed(int2 pixel, int width, int index, int dimension)
	{
		seed = pixel.x + pixel.y * width;
		seed *= index;
		seed += dimension;
	}
	__device__ float IndependentSampler::get1D()
	{
		seed = rng.pcg_hash(seed);
		return (float)seed / (float)UINT32_MAX;
	}
	__device__ float2 IndependentSampler::get2D()
	{
		return make_float2(get1D(), get1D());
	}
	__device__ float3 sampleUniformSphere(float2 xi)
	{
		float z = 1 - 2 * xi.x;
		float r = fmaxf(0, sqrtf(1 - (z * z)));
		float phi = 2 * Constants::PI * xi.y;
		return { r * cosf(phi), r * sinf(phi), z };
	}

	__device__ float3 sampleCosineWeightedHemisphere(float2 xi)
	{
		// Generate a cosine-weighted direction in the local frame
		float phi = 2.0f * Constants::PI * xi.x;
		float cosTheta = sqrtf(xi.y);
		float sinTheta = sqrtf(1.0f - xi.y);

		float3 H;
		H.x = sinTheta * cosf(phi);
		H.y = sinTheta * sinf(phi);
		H.z = cosTheta;

		return H;
	}
}