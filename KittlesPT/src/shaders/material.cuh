#pragma once
#include <vector_types.h>

struct Material
{
	__device__ __host__ Material(float3 albedo, float roughness) :
		albedo(albedo), roughness(roughness) {}
	float3 albedo;
	float roughness;
};