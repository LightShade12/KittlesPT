#pragma once
#include <vector_types.h>
#include <cstdint>

namespace KittlesPT
{
	struct ShaderData;
	class RGBSpectrum;
	class Ray;
	class BSDF;
	class Light;

	struct SurfaceInteraction
	{
		__device__ RGBSpectrum Le(const ShaderData& shader_data, const Ray& ray) const;

		__device__ BSDF getBSDF(const ShaderData& shader_data) const;

		__device__ void skipInteraction(Ray* ray);

		__device__ Ray spawnRay(float3 wi, int scatter_flags) const;

		__device__ Ray spawnRayTo(float3 target) const;

		//--------------------------------------------------
		float3 wo{ 0.0f,0.0f,0.0f };
		const Light* arealight = nullptr;
		int32_t material_id = -1;
		float distance = INFINITY;
		float2 uv{ 0.0f,0.0f };
		float3 world_position{ 0.0f,0.0f,0.0f };
		float3 world_geometric_normal{ 0.0f,0.0f,0.0f };
		bool backface = false;
	};

	struct Intersection
	{
		//closest hit shader
		__device__ SurfaceInteraction getSurfaceInteraction(const ShaderData& shader_data, const Ray& ray);

		__device__ bool operator!()
		{
			return (primitive_id < 0);
		}
		__device__ operator bool()
		{
			return (primitive_id >= 0);
		}

		//-------------------------------------
		float distance = INFINITY;
		int32_t primitive_id = -1;
		int32_t instance_id = -1;
		float3 bary_coords{ 0.0f,0.0f,0.0f };
	};
}/*KittlesPT*/