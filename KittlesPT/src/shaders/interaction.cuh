#pragma once
#include <vector_types.h>

namespace KittlesPT
{
	struct GlobalShaderData;
	class RGBSpectrum;
	class Ray;
	class BSDF;
	class Light;

	struct SurfaceInteraction
	{
		__device__ RGBSpectrum Le(const GlobalShaderData& shader_data, const Ray& ray) const;

		__device__ BSDF getBSDF(const GlobalShaderData& shader_data) const;

		__device__ void skipInteraction(Ray* ray);

		__device__ Ray spawnRay(float3 wi, int scatter_flags) const;

		__device__ Ray spawnRayTo(float3 target) const;

		//--------------------------------------------------
		float distance = -1;
		float2 uv;
		float3 world_position;
		float3 world_geometric_normal;
		int material_id = -1;
		bool backface = false;
		const Light* light = nullptr;
	};

	struct Intersection
	{
		//closest hit shader
		__device__ SurfaceInteraction getSurfaceInteraction(const GlobalShaderData& shader_data, const Ray& ray);

		__device__ bool operator ! ();

		//--------------------------------------------------------
		float distance = -1;
		int instance_id = -1;
	};
}/*KittlesPT*/