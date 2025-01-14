#pragma once
#include <vector_types.h>

namespace KittlesPT
{
	struct FilterSample
	{
		__device__ FilterSample(float2 p, float w) :
			p(p), weight(w) {}
		float2 p{};//point
		float weight = 1;
	};

	class BoxFilter
	{
	public:
		__device__ BoxFilter(float2 radius) :
			m_radius(radius) {}

		__device__ float evaluate(float2 p) const;

		__device__ FilterSample sample(float2 u) const;

		__device__ float integral() const;

	private:
		float2 m_radius{ 1.0f,1.0f };
	};

	/*
	* class Filter
	{
	public:

		__device__ Filter(float2 radius, float alpha = 0.5f) :
			radius(radius),
			alpha(alpha),
			expX(exp(-alpha * radius.x * radius.x)),
			expY(exp(-alpha * radius.y * radius.y))
		{};

		__device__ float evaluate(float2 p) const
		{
			//return (fmaxf(0.0f, Gaussian(p.x, 0, alpha) - expX) * fmaxf(0.0f, Gaussian(p.y, 0, alpha) - expY));
			return Gaussian(p.x, expX) * Gaussian(p.y, expY);
			//return (fabsf(p.x) <= radius.x && fabsf(p.y) <= radius.y) ? 1.0f : 0.0f;
		}

		__device__ FilterSample sample(float2 u) const
		{
			float2 p = make_float2(lerp(-radius.x, radius.x, u.x), lerp(-radius.y, radius.y, u.y));
			//float2 p = make_float2(sampleTent(u.x, radius.x), sampleTent(u.y, radius.y));
			float w = evaluate(p);
			return FilterSample(p, w);
		}

		//TODO:wrong integral
		__device__ float integral()
		{
			return 2.0f * radius.x * 2.0f * radius.y;
		}

		__device__ float Gaussian(float d, float expv) const {
			return fmaxf(0.0f, exp(-alpha * d * d) - expv);
		}

		float alpha = 0.5f;
		float expX, expY;
		float2 radius = make_float2(1.0f);
	};
	*/
}/*KittlesPT*/