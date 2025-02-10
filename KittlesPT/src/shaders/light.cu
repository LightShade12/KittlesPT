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
		float dist = distance(confirmed_ls.wpos_light, ctx.w_pos);
		float cos_theta_L = AbsDot(normalize(confirmed_ls.wpos_light - ctx.w_pos), confirmed_ls.wgnorm);
		float pdf = area_pdf / (cos_theta_L / Sqr(dist));
		return pdf;
	}

	__device__ RGBSpectrum Light::L(const ShaderData& shader_data, float2 uv) const
	{
		Material mat = shader_data.materials_buffer.data[shader_data.triangles_buffer.data[prim_id].material_id];
		RGBSpectrum emission = RGBSpectrum(mat.emissive_factor * mat.emission_scale_nits);
		if (mat.emission_texture_id >= 0) {
			TextureEvalContext ctx({}, uv);
			RGBSpectrum sampled = shader_data.texture_buffer.data[mat.emission_texture_id].evaluate(shader_data, ctx);
			sampled = sampled.gamma2_2ToLinear();//sRGB to linear approx
			emission *= sampled;
		}
		return emission;
	};

	__device__ LightLiSample Light::sampleLi(const ShaderData& shader_data, const LightSampleContext& ctx, float2 u2) const
	{
		ShapeSample ss = shader_data.triangles_buffer.data[prim_id].sample(u2, ctx);

		float3 p = ss.wgnorm;
		float theta = acosf(-p.y); float phi = atan2(-p.z, p.x) + Constants::PI;
		float2 uv = make_float2(phi / (2.0f * Constants::PI), theta / Constants::PI);

		float3 wi = normalize(ss.wpos - ctx.w_pos);
		//only for full sphere sampling
		//ss.pdf /= (AbsDot(-wi, ss.wgnorm) / Sqr(length(ss.wpos - ctx.w_pos)));//conversion to solid angle

		RGBSpectrum Le = L(shader_data, uv);

		return LightLiSample(Le, wi, ss.wpos, ss.wgnorm, ss.pdf);
	}
}/*KittlesPT*/