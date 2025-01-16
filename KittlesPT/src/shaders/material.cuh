#pragma once
#include <vector_types.h>

namespace KittlesPT
{
	class BSDF;
	struct GlobalShaderData;
	struct SurfaceInteraction;

	struct MaterialEvalContext
	{
		__device__ explicit MaterialEvalContext(const SurfaceInteraction& surf);

		float2 uv;
		float3 wpos;
		float3 wgnorm;
		bool backface = false;
	};

	struct Material
	{
		__device__ __host__ Material(
			float3 albedo,
			float metallicity,
			float roughness,
			float transmission,
			float ior,
			float3 emission_factor,
			float emission_scale,
			int albedo_texture_id) :
			albedo(albedo),
			metallicity(metallicity),
			roughness(roughness),
			transmission(transmission),
			ior(ior),
			emissive_factor(emission_factor),
			emission_scale(emission_scale),
			albedo_texture_id(albedo_texture_id)
		{}

		__device__ BSDF getBSDF(const GlobalShaderData& shader_data, MaterialEvalContext ctx) const;

		//----
		float3 albedo{ 0.8f,0.8f,0.8f };
		int albedo_texture_id = -1;
		float metallicity = 0.0f;
		float roughness = 0.5f;
		float transmission = 0.0f;
		float ior = 1.45f;
		float3 emissive_factor{ 0.0f,0.0f,0.0f };
		float emission_scale = 1.0f;//unit: nit(cd/m2) qty name: luminance
	};
}/*KittlesPT*/