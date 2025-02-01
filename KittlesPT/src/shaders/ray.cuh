#pragma once
#include "maths/linear_algebra.cuh"

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
		__device__ void setOrigin(float3 origin) { m_origin = origin; }
		__device__ float3 getDirection() const { return m_direction; }
		__device__ void setDirection(float3 direction) {
			m_direction = direction;
			m_inv_direction = 1.0f / m_direction;
		}
		__device__ float3 getInvDirection() const { return m_inv_direction; }
		__device__ float3 getPointAt(float dist) const { return m_origin + (dist * m_direction); };

		//TODO: ensure normality
		__device__ Ray transform(const Mat4& basis) const {
			return Ray(make_float3(basis * make_float4(m_origin, 1)),
				make_float3(basis * make_float4(m_direction, 0)));
		}
	private:
		float3 m_origin{ 0.0f,0.0f,0.0f };
		float3 m_direction{ 0.0f,0.0f,0.0f };
		float3 m_inv_direction{ 0.0f,0.0f,0.0f };
	};
}/*KittlesPT*/