#include "filter.cuh"
#include "maths/vector_maths.cuh"

namespace KittlesPT
{
	__device__ float BoxFilter::evaluate(float2 p) const
	{
		return (fabsf(p.x) <= m_radius.x && fabsf(p.y) <= m_radius.y) ? 1.0f : 0.0f;
	}

	__device__ FilterSample BoxFilter::sample(float2 u) const
	{
		float2 p = make_float2(::lerp(-m_radius.x, m_radius.x, u.x), ::lerp(-m_radius.y, m_radius.y, u.y));
		float w = 1.0f;
		return FilterSample(p, w);
	}

	__device__ float BoxFilter::integral() const
	{
		return 2.0f * m_radius.x * 2.0f * m_radius.y;
	}
}/*KittlesPT*/