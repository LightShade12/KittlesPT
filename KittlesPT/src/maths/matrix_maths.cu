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
}