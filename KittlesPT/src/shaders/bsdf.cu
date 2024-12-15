#include "bsdf.cuh"
#include "samplers.cuh"

namespace KittlesPT
{
	BSDF::BSDF(const Mat3& tangent_basis, float3 albedo) :
		albedo_factor(albedo)
	{
		tangent_matrix = tangent_basis;
	}

	__device__ float3 BSDF::f(float3 w_wo, float3 w_wi) const
	{
		float3 wo = tangent_matrix.inverse() * w_wo;
		//float3 wo = w_wo;
		float3 wi = tangent_matrix.inverse() * w_wi;

		return fOpaqueDielectric(wo, wi);
	}

	__device__ float BSDF::pdf(float3 w_wo, float3 w_wi) const
	{
		float3 wi = tangent_matrix.inverse() * w_wi;
		//float3 wo = w_wo;
		float3 wo = tangent_matrix.inverse() * w_wo;
		return pdfOpaqueDielectric(wo, wi);
	}

	__device__ BSDFSample BSDF::sampleBSDF(float3 w_wo, float2 u2) const
	{
		float3 wo = tangent_matrix.inverse() * w_wo;
		BSDFSample bs = sampleOpaqueDielectric(wo, u2);
		bs.wi = tangent_matrix * bs.wi;
		return bs;
	}

	//=========================================================================================

	//===========================================================================================================
	//DIFFUSE MATERIAL BRDF
	//===========================================================================================================

	__device__ float3 BSDF::sampleDiffuseBRDF(float2 u2) const
	{
		return sampleCosineWeightedHemisphere(u2);
	}

	__device__ float BSDF::fDiffuseBRDF(float3 wo, float3 wi) const
	{
		float scalar_switch = (wo.z * wi.z > 0) ? 1 : 0;
		float out = (scalar_switch / Constants::PI);
		return out;
	}

	__device__ float BSDF::pdfDiffuseBRDF(float3 wo, float3 wi) const
	{
		return wi.z / Constants::PI;
	}

	//===========================================================================================================
	//OPAQUE DIELECTRIC BSDF
	//===========================================================================================================

	__device__ BSDFSample BSDF::sampleOpaqueDielectric(float3 wo, float2 u2) const
	{
		float3 wi = sampleDiffuseBRDF(u2);
		float3 f = albedo_factor * fDiffuseBRDF(wo, wi);
		float pdf = pdfDiffuseBRDF(wo, wi);

		return BSDFSample(f, wi, pdf);
	}

	__device__ float3 BSDF::fOpaqueDielectric(float3 wo, float3 wi) const
	{
		return albedo_factor * fDiffuseBRDF(wo, wi);
	}

	__device__ float BSDF::pdfOpaqueDielectric(float3 wo, float3 wi) const
	{
		return pdfDiffuseBRDF(wo, wi);
	}
}