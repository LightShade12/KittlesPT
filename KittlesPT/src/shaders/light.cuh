#pragma once
#include "color.cuh"
#include <vector_types.h>
#include <cuda_runtime.h>

namespace KittlesPT
{
	class Sphere;
	struct SurfaceInteraction;
	struct GlobalShaderData;

	//Represents a sampled point on light
	struct LightLiSample
	{
		__device__ LightLiSample() = default;

		__device__ LightLiSample(
			const RGBSpectrum& L, const float3& wi, const float3& pLight,
			const float3& gwn, float pdf)
			: L(L), wi(wi), wpos_light(pLight), geo_wnorm(gwn), pdf(pdf)
		{}

		//---------------------------------------

		__device__ bool operator !()
		{
			return (pdf <= 0);
		}

		//---------------------------------------

		//TODO: can add float2 uv, float3 wo via Interaction struct
		float3 wpos_light;
		float3 geo_wnorm;

		RGBSpectrum L;
		float3 wi;
		float pdf = 0;
	};

	//===================================================================================================

	//Surface data type; passed to sampler
	struct LightSampleContext
	{
		__device__ LightSampleContext() = default;

		__device__ LightSampleContext(const SurfaceInteraction& si);

		//---------------------
		float3 w_pos{};
		float3 geo_wnorm{};
		float3 s_wnorm{};
	};

	//=====================================================================================================

	//The Light object that is bound to a primitive
	class Light
	{
	public:
		__host__ __device__ Light(Sphere* primitive, int prim_id, float3 color, float power);

		//----------------------------------------------------------------------------

		__device__ RGBSpectrum L(float3 p, float3 n, float3 wi) const;

		__device__ LightLiSample sampleLi(const GlobalShaderData& shader_data, const LightSampleContext& ctx, float2 u2) const;

		__device__ float pdf_Li(const LightSampleContext& ctx, float3 wo, float3 confirmed_hit_wpos, float3 confirmed_hit_geo_norm) const;

	private:

		int prim_id = -1;
		RGBSpectrum L_emit;
		float emission_scale;
		float area;
	};
}