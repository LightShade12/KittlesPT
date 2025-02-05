#pragma once
#include "color.cuh"
#include "interaction.cuh"
#include "maths/constants.cuh"
#include "material.cuh"
#include "texture.cuh"

#include <vector_types.h>
#include <cuda_runtime.h>

namespace KittlesPT
{
	class Sphere;
	struct ShaderData;
	//Surface data type; passed to sampler
	struct LightSampleContext
	{
		LightSampleContext() = default;

		explicit __device__ LightSampleContext(const SurfaceInteraction& si) :
			w_pos(si.world_position), wgnorm(si.world_geometric_normal)
		{};

		//----------------------------------

		float3 w_pos{};
		float3 wgnorm{};
		float3 s_wnorm{};
	};

	//=====================================================================================================

	//Represents a sampled point on light
	struct LightLiSample
	{
		LightLiSample() = default;

		__device__ LightLiSample(
			const RGBSpectrum& L, const float3& wi, const float3& pLight,
			const float3& gwn, float pdf)
			: L(L), wi(wi), wpos_light(pLight), wgnorm(gwn), pdf(pdf)
		{}

		__device__ explicit LightLiSample(const SurfaceInteraction& surf)
			: wpos_light(surf.world_position), wgnorm(surf.world_geometric_normal), L(0.0f)
		{}

		//op---------------------------------------

		__device__ bool operator !()
		{
			return (pdf <= 0 || !L);
		}

		//---------------------------------------

		//TODO: can add float2 uv, float3 wo via Interaction struct
		float3 wpos_light{ 0.0f,0.0f,0.0f };
		float3 wgnorm{ 0.0f,0.0f,0.0f };

		RGBSpectrum L;
		float3 wi{ 0.0f,0.0f,0.0f };
		float pdf = 0;
	};

	//===================================================================================================

	//The Light object that is bound to a primitive
	class Light
	{
	public:
		__host__ Light(float area, int prim_id, float3 emission_color, float emission_nits) :
			emission_spectrum_rgb(emission_color), emission_nits(emission_nits), prim_id(prim_id), area(area) {};

		//----------------------------------------------------------------------------

		__device__ RGBSpectrum L(const ShaderData& shader_data, float2 uv) const;

		__device__ LightLiSample sampleLi(const ShaderData& shader_data, const LightSampleContext& ctx, float2 u2) const;

		//TODO: maybe consider allowing this method to test intersection on its shape for bug free, reliable operation
		__device__ float pdf_Li(const LightSampleContext& ctx, const LightLiSample& confirmed_ls) const;

		//Net power
		__device__ float phi()
		{
			return (Constants::PI * 2.0f * area * emission_spectrum_rgb * emission_nits);
		}

	public:
		int prim_id = -1;
	private:
		RGBSpectrum emission_spectrum_rgb;
		float emission_nits = 0.0f;
		float area = 0.0f;
	};
}/*KittlesPT*/