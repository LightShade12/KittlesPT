#include "linear_algebra.cuh"
#include "constants.cuh"
#include "shaders/packing.cuh"

namespace KittlesPT
{
	__host__ __device__ float deg2rad(float degree)
	{
		return (degree * (Constants::PI / 180.f));
	}

	__host__ __device__ float Sqr(float v)
	{
		return v * v;
	}

	__device__ float Gaussian(float x, float mu, float sigma)
	{
		return (1.0f / sqrtf(2 * Constants::PI * sigma * sigma) *
			exp(-Sqr(x - mu) / (2 * sigma * sigma)));
	}

	__constant__ constexpr float one_minus_epsilon = 0x1.fffffep-1;

	__device__ float sampleLinear(float u, float a, float b)
	{
		if (u == 0 && a == 0) {
			return 0;
		}
		float x = u * (a + b) / (a + sqrtf(lerp(Sqr(a), Sqr(b), u)));
		return fminf(x, one_minus_epsilon);
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
			*uRemapped = fminf((up - sum) / weights[offset], one_minus_epsilon);
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
}