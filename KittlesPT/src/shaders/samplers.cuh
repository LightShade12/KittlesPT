#pragma once
#include "maths/constants.cuh"
#include "maths/linear_algebra.cuh"
#include <cuda_runtime.h>
#include <cstdint>

namespace KittlesPT
{
	class RNG
	{
	public:
		inline __device__ uint32_t pcg_hash(uint32_t input)
		{
			uint32_t state = input * 747796405u + 2891336453u;
			uint32_t word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
			return (word >> 22u) ^ word;
		}
	};

	class IndependentSampler
	{
	public:

		inline __device__ void initPixelSeed(int2 pixel, int width, int index, int dimension = 0)
		{
			m_seed = pixel.x + pixel.y * width;
			m_seed *= (index == 0) ? 1.0f : index;
			m_seed += dimension;
		}

		inline __device__ float get1D()
		{
			m_seed = m_rng.pcg_hash(m_seed);
			return (float)m_seed * Constants::INV_UINT32MAX;
		}

		__device__ float2 get2D()
		{
			return make_float2(get1D(), get1D());
		}

		__device__ uint32_t getSeed() { return m_seed; }
		__device__ void setSeed(uint32_t v) { m_seed = v; }

	private:
		uint32_t m_seed;
		RNG m_rng;
	};

	inline __device__ float3 sampleUniformTriangle(float2 u)
	{
		float2 bary;
		if (u.x < u.y) {
			bary.x = u.x / 2.0f;
			bary.y = u.y - bary.x;
		}
		else {
			bary.y = u.y / 2.0f;
			bary.x = u.x - bary.y;
		}
		return { bary.x, bary.y, 1.0f - bary.x - bary.y };
	}

	//TODO: learn more about these
	//NOTE: All this sampling is done in tangent space
	inline __device__ float2 sampleUniformDiskPolar(float2 u)
	{
		float r = sqrtf(u.x);
		float theta = 2.0f * Constants::PI * u.y;
		return make_float2(r * cosf(theta), r * sinf(theta));
	};

	inline __device__ float3 sampleUniformHemisphere(float2 u)
	{
		float z = u.x;
		float r = sqrtf(1.0f - Sqr(z));
		float phi = 2.0f * Constants::PI * u.y;
		return  make_float3(r * cosf(phi), r * sinf(phi), z);
	}
	inline __device__ float3 sampleUniformSphere(float2 xi)
	{
		float z = 1.0f - 2.0f * xi.x;
		float r = fmaxf(0, sqrtf(1 - (z * z)));
		float phi = 2.0f * Constants::PI * xi.y;
		return make_float3(r * cosf(phi), r * sinf(phi), z);
	}

	inline __device__ float3 sampleCosineWeightedHemisphere(float2 xi)
	{
		float phi = 2.0f * Constants::PI * xi.x;
		float cos_theta = sqrtf(xi.y);
		float sin_theta = sqrtf(1.0f - xi.y);

		return make_float3(sin_theta * cosf(phi), sin_theta * sinf(phi), cos_theta);
	}

	inline __device__ float3 toSphericalDirection(float sin_theta, float cos_theta, float phi)
	{
		return make_float3(
			clamp(sin_theta, -1.0f, 1.0f) * cosf(phi),
			clamp(sin_theta, -1.0f, 1.0f) * sinf(phi),
			clamp(cos_theta, -1.0f, 1.0f));
	}

	inline __device__ float3 toSphericalDirection(float theta, float phi)
	{
		return make_float3(
			clamp(sinf(theta), -1.0f, 1.0f) * cosf(phi),
			clamp(sinf(theta), -1.0f, 1.0f) * sinf(phi),
			clamp(cosf(theta), -1.0f, 1.0f));
	}

	inline __device__ float3 sampleUniformCone(float2 u, float cos_theta_max)
	{
		float cos_theta = (1.0f - u.x) + u.x * cos_theta_max;
		float sin_theta = sqrtf(1.0f - Sqr(cos_theta));
		float phi = u.y * 2.0f * Constants::PI;
		return toSphericalDirection(sin_theta, cos_theta, phi);
	}
}/*KittlesPT*/