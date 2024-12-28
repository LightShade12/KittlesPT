#include "samplers.cuh"
#include "maths/constants.cuh"
#include "maths/linear_algebra.cuh"

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
	__device__ float2 sampleUniformDiskPolar(float2 u)
	{
		float r = sqrtf(u.x);
		float theta = 2 * Constants::PI * u.y;
		return make_float2(r * cosf(theta), r * sinf(theta));
	};

	__device__ float3 sampleUniformHemisphere(float2 u)
	{
		float z = u.x;
		float r = sqrtf(1 - Sqr(z));
		float phi = 2 * Constants::PI * u.y;
		return  make_float3(r * cosf(phi), r * sinf(phi), z);
	}
	__device__ float3 sampleUniformSphere(float2 xi)
	{
		float z = 1 - 2 * xi.x;
		float r = fmaxf(0, sqrtf(1 - (z * z)));
		float phi = 2 * Constants::PI * xi.y;
		return make_float3(r * cosf(phi), r * sinf(phi), z);
	}

	__device__ float3 sampleCosineWeightedHemisphere(float2 xi)
	{
		float phi = 2.0f * Constants::PI * xi.x;
		float cos_theta = sqrtf(xi.y);
		float sin_theta = sqrtf(1.0f - xi.y);

		return make_float3(sin_theta * cosf(phi), sin_theta * sinf(phi), cos_theta);
	}

	__device__ float3 toSphericalDirection(float sin_theta, float cos_theta, float phi)
	{
		return make_float3(
			clamp(sin_theta, -1.0f, 1.0f) * cosf(phi),
			clamp(sin_theta, -1.0f, 1.0f) * sinf(phi),
			clamp(cos_theta, -1.0f, 1.0f));
	}

	__device__ float3 toSphericalDirection(float theta, float phi)
	{
		return make_float3(
			clamp(sinf(theta), -1.0f, 1.0f) * cosf(phi),
			clamp(sinf(theta), -1.0f, 1.0f) * sinf(phi),
			clamp(cosf(theta), -1.0f, 1.0f));
	}

	__device__ float3 sampleUniformCone(float2 u, float cos_theta_max)
	{
		float cos_theta = (1 - u.x) + u.x * cos_theta_max;
		float sin_theta = sqrtf(1 - Sqr(cos_theta));
		float phi = u.y * 2 * Constants::PI;
		return toSphericalDirection(sin_theta, cos_theta, phi);
	}
}