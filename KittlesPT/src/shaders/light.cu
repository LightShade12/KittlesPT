#include "light.cuh"

#include "maths/linear_algebra.cuh"
#include "sphere.cuh"
#include "containers.cuh"

namespace KittlesPT
{
	//LIGHT===============================================================================

	__device__ float Light::pdf_Li(const LightSampleContext& ctx, const LightLiSample& confirmed_ls) const
	{
		float area_pdf = 1.0f / area;
		float dist = length(confirmed_ls.wpos_light - ctx.w_pos);
		float cos_theta_L = AbsDot(normalize(confirmed_ls.wpos_light - ctx.w_pos), confirmed_ls.wgnorm);
		float pdf = area_pdf / (cos_theta_L / Sqr(dist));
		return pdf;
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
}/*KittlesPT*/