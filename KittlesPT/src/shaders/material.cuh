#pragma once
#include "maths/vector_maths.cuh"

namespace KittlesPT
{
	struct Material
	{
		__device__ __host__ Material(
			float3 albedo,
			float metallicity,
			float roughness,
			float transmission,
			float ior) :
			albedo(albedo),
			metallicity(metallicity),
			roughness(roughness),
			transmission(transmission),
			ior(ior)
		{}

		//----
		float3 albedo = make_float3(0.8);
		float metallicity = 0.0f;
		float roughness = 0.5f;
		float transmission = 0.0f;
		float ior = 1.45f;
	};
}