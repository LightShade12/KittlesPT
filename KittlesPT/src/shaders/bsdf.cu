#include "bsdf.cuh"
#include "samplers.cuh"

namespace KittlesPT
{
	BSDF::BSDF(const Mat3& tangent_matrix_, float3 albedo)
	{
		tangent_matrix = tangent_matrix_;
		albedo_factor = albedo;
	}

	__device__ float3 BSDF::f(float3 r_wo, float3 r_wi) const
	{
		//float3 wo = tangentMatrix.inverse() * r_wo;
		float3 wi = tangent_matrix.inverse() * r_wi;
		float3 wo = r_wo;
		//float3 wi = r_wi;

		return fOpaqueDielectric(wo, wi);
	}

	__device__ float BSDF::pdf(float3 r_wo, float3 r_wi) const
	{
		float3 wi = tangent_matrix.inverse() * r_wi;
		float3 wo = r_wo;
		return pdfOpaqueDielectric(wo, wi);
	}

	__device__ BSDFSample BSDF::sampleBSDF(float3 r_wo, float2 u2) const
	{
		float3 wo = tangent_matrix.inverse() * r_wo;
		BSDFSample bs = sampleOpaqueDielectric(wo, u2);
		bs.wi = tangent_matrix * bs.wi;
		return bs;
	}

	__device__ BSDFSample BSDF::sampleOpaqueDielectric(float3 wo, float2 u2) const
	{
		float3 wi = sampleCosineWeightedHemisphere(u2);
		float3 f = fOpaqueDielectric(wo, wi);
		float pdf = pdfOpaqueDielectric(wo, wi);
		return BSDFSample(f, wi, pdf);
	}

	__device__ float3 BSDF::fOpaqueDielectric(float3 wo, float3 wi) const
	{
		return (albedo_factor / Constants::PI);
		//return (float3(0.8) / PI);
	}

	__device__ float BSDF::pdfOpaqueDielectric(float3 wo, float3 wi) const
	{
		return AbsDot({ 0,0,1 }, wi) / Constants::PI;
	}
}