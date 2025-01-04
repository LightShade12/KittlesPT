#pragma once
#include "maths/linear_algebra.cuh"
#include "samplers.cuh"

namespace KittlesPT
{
	struct FilterSample
	{
		FilterSample() = default;
		__device__ FilterSample(float2 p, float w) :
			p(p), weight(w) {}
		float2 p{};
		float weight = 1;
	};

	//Box Filter
	class Filter
	{
	public:
		Filter() = default;

		__device__ float evaluate(float2 p) const
		{
			return (fabsf(p.x) <= radius.x && fabsf(p.y) <= radius.y) ? 1.0f : 0.0f;
		}

		__device__ FilterSample sample(float2 u) const
		{
			float2 p = make_float2(lerp(-radius.x, radius.x, u.x), lerp(-radius.y, radius.y, u.y));
			float w = evaluate(p);
			return FilterSample(p, w);
		}

		__device__ float integral()
		{
			return 2.0f * radius.x * 2.0f * radius.y;
		}

		float2 radius = make_float2(1.0f);
	};
}