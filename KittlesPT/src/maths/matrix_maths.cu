#include "matrix_maths.cuh"

namespace KittlesPT
{
	__device__ Mat3 generateOrthonormalBasis(const float3& normal)
	{
		// Choose a helper vector H that is not parallel to the normal
		float3 helper = (fabs(normal.x) > fabs(normal.z)) ? make_float3(0, 1, 0) : make_float3(1, 0, 0);

		// Compute tangent vector (orthogonal to normal)
		float3 tangent = normalize(cross(helper, normal));

		// Compute bitangent vector (orthogonal to both normal and tangent)
		float3 bitangent = cross(normal, tangent);

		return Mat3(tangent, bitangent, normal);
	}

	__device__ Mat3 generateONBFrisvad(float3 normal)
	{
		Mat3 ret;
		ret[1] = normal;
		if (normal.z < -0.999805696f)
		{
			ret[0] = make_float3(0.0f, -1.0f, 0.0f);
			ret[2] = make_float3(-1.0f, 0.0f, 0.0f);
		}
		else
		{
			float a = 1.0f / (1.0f + normal.z);
			float b = -normal.x * normal.y * a;
			ret[0] = make_float3(1.0f - normal.x * normal.x * a, b, -normal.x);
			ret[2] = make_float3(b, 1.0f - normal.y * normal.y * a, -normal.y);
		}

		ret[1] = ret[2];
		ret[2] = normal;

		return ret;
	}
}