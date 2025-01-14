#pragma once
#include "maths/vector_maths.cuh"

namespace KittlesPT
{
	class Ray
	{
	public:
		__device__ Ray(float3 orig, float3 dir) :
			m_origin(orig),
			m_direction(dir),
			m_inv_direction(1.0f / dir) {};
		__device__ float3 getOrigin() const { return m_origin; }
		__device__ float3 getDirection() const { return m_direction; }
		__device__ float3 getInvDirection() const { return m_inv_direction; }
		__device__ float3 getPointAt(float dist) const { return m_origin + (dist * m_direction); };
	private:
		float3 m_origin{ 0.0f,0.0f,0.0f };
		float3 m_direction{ 0.0f,0.0f,0.0f };
		float3 m_inv_direction{ 0.0f,0.0f,0.0f };
	};
}/*KittlesPT*/