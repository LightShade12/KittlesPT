#pragma once
#include <vector_types.h>

namespace KittlesPT
{
	class UnifiedBSDF;
	struct ShaderData;
	struct SurfaceInteraction;

	struct MaterialEvalContext
	{
		__device__ explicit MaterialEvalContext(const SurfaceInteraction& surface);

		float3 wo;
		float2 uv;
		float3 wpos;
		float3 wgnorm;
		bool backface = false;
	};

	struct Material
	{
		__device__ __host__ Material(
			int albedo_texture_id,
			float3 albedo,
			int orm_texture_id,
			float metallicity,
			float roughness,
			int transmission_texture_id,
			float transmission,
			float ior,
			int emission_texture_id,
			float3 emission_factor,
			float emission_scale,
			int normal_texture_id,
			float normal_scale
		) :
			albedo_texture_id(albedo_texture_id),
			albedo(albedo),
			ORM_texture_id(orm_texture_id),
			metallic_factor(metallicity),
			roughness_factor(roughness),
			transmission_texture_id(transmission_texture_id),
			transmission_factor(transmission),
			ior(ior),
			emission_texture_id(emission_texture_id),
			emissive_factor(emission_factor),
			emission_scale_nits(emission_scale),
			normal_texture_id(normal_texture_id),
			normal_scale(normal_scale)
		{}

		__device__ UnifiedBSDF getBSDF(const ShaderData& shader_data, MaterialEvalContext ctx) const;

		//----
	public:
		int albedo_texture_id = -1;
		float3 albedo{ 0.8f,0.8f,0.8f };

		int ORM_texture_id = -1;
		float metallic_factor = 0.0f;
		float roughness_factor = 0.5f;

		int transmission_texture_id = -1;
		float transmission_factor = 0.0f;

		float ior = 1.45f;

		int emission_texture_id = -1;
		float3 emissive_factor{ 0.0f,0.0f,0.0f };
		float emission_scale_nits = 1.0f;//unit: nit(cd/m2) qty name:luminance

		int normal_texture_id = -1;
		float normal_scale = 1.0f;
	};
}/*KittlesPT*/