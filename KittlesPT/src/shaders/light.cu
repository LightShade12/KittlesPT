#include "light.cuh"

#include "maths/linear_algebra.cuh"
#include "maths/constants.cuh"
#include "interaction.cuh"
#include "sphere.cuh"
#include "containers.cuh"

namespace KittlesPT
{
	//LIGHT SAMPLE CONTEXT===============================================================

	__device__ LightSampleContext::LightSampleContext(const SurfaceInteraction& si) :
		w_pos(si.world_position), wgnorm(si.world_geometric_normal)
	{};

	//LIGHT LI SAMPLE====================================================================

	__device__ LightLiSample::LightLiSample(const SurfaceInteraction& surf)
		:wpos_light(surf.world_position), wgnorm(surf.world_geometric_normal), L(0.0f)
	{}

	//LIGHT===============================================================================

	__device__ RGBSpectrum Light::L(float3 p, float3 n, float3 wi) const
	{
		return L_emit * emission_scale;
	};

	__device__ float Light::pdf_Li(const LightSampleContext& ctx, const LightLiSample& confirmed_ls) const
	{
		float area_pdf = 1.0f / area;
		float dist = length(confirmed_ls.wpos_light - ctx.w_pos);
		float cos_theta_L = AbsDot(normalize(confirmed_ls.wpos_light - ctx.w_pos), confirmed_ls.wgnorm);
		float pdf = area_pdf / (cos_theta_L / Sqr(dist));
		return pdf;
	}

	__device__ float Light::phi()
	{
		return (Constants::PI * 2.0f * area * L_emit * emission_scale);
	}

	__device__ LightLiSample Light::sampleLi(const GlobalShaderData& shader_data, const LightSampleContext& ctx, float2 u2) const
	{
		ShapeSample ss = shader_data.geometry_buffer.data[prim_id].sample(u2);

		float3 wi = normalize(ss.wpos - ctx.w_pos);
		//only for full sphere sampling
		ss.pdf /= (AbsDot(-wi, ss.wgnorm) / Sqr(length(ss.wpos - ctx.w_pos)));//conversion to solid angle

		RGBSpectrum Le = L(ss.wpos, ss.wgnorm, wi);

		return LightLiSample(Le, wi, ss.wpos, ss.wgnorm, ss.pdf);
	}
}