#include "light.cuh"
#include "maths/linear_algebra.cuh"
#include "sphere.cuh"
#include "containers.cuh"

namespace KittlesPT
{
	__device__ LightSampleContext::LightSampleContext(const SurfaceInteraction& si)
	{
		w_pos = si.world_position;
		geo_wnorm = si.world_geometric_normal;
	};

	__host__ __device__ Light::Light(Sphere* primitive, int prim_id, float3 color, float power)
		:L_emit(color), emission_scale(power), prim_id(prim_id)
	{
		area = primitive->getProjectedArea();
	}
	__device__ RGBSpectrum Light::L(float3 p, float3 n, float3 wi) const
	{
		return L_emit * emission_scale;
	};
	__device__ float Light::pdf_Li(const LightSampleContext& ctx, float3 wo, float3 confirmed_hit_wpos, float3 confirmed_hit_geo_norm) const
	{
		float area_pdf = 1.f / area;
		float dist = length(ctx.w_pos - confirmed_hit_wpos);
		float cosTheta_emitter = AbsDot(wo, confirmed_hit_geo_norm);
		float pdf = area_pdf * (1.f / cosTheta_emitter) * Sqr(dist);
		return pdf;
	}

	__device__ LightLiSample Light::sampleLi(const GlobalShaderData& shader_data, const LightSampleContext& ctx, float2 u2) const
	{
		ShapeSample ss = shader_data.geometry_buffer.data[prim_id].sample(u2);

		float3 wi = normalize(ss.point - ctx.w_pos);
		RGBSpectrum Le = L(ss.point, ss.geo_w_normal, wi);

		return LightLiSample(Le, wi, ss.point, ss.geo_w_normal, ss.pdf);
	}
}