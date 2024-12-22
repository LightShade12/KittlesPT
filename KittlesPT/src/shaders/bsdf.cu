#include "bsdf.cuh"
#include "samplers.cuh"
#include "../maths/linear_algebra.cuh"

namespace KittlesPT
{
	BSDF::BSDF(const Mat3& tangent_basis,
		float3 albedo,
		float metallicity,
		float roughness,
		float transmission) :
		albedo_factor(albedo),
		metallicity(metallicity),
		roughness(roughness),
		transmission(transmission)
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

	__device__ BSDFSample BSDF::sampleBSDF(float3 w_wo, float2 u2, float2 X2) const
	{
		float path_probability = X2.x;
		float3 wo = tangent_matrix.inverse() * w_wo;

		float prob_metallic = metallicity;

		BSDFSample bs;

		if (path_probability < prob_metallic)
		{
			bs = sampleConductor(wo, u2, X2.y);
		}
		else
		{
			bs = sampleOpaqueDielectric(wo, u2, X2.y);
		}

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
	//GLOSSY MICROFACET BRDF
	//===========================================================================================================
	//SmithGGXMaskingShadowing
	__device__ float G2_Smith(float3 wo, float3 wi, float roughness)
	{
		float a2 = ::powf(roughness, 4);

		float dotNL = fabs(wi.z);
		float dotNV = fmaxf(0, wo.z);

		float denomA = dotNV * sqrtf(a2 + (1.0f - a2) * Sqr(dotNL));
		float denomB = dotNL * sqrtf(a2 + (1.0f - a2) * Sqr(dotNV));

		return 2.0f * dotNL * dotNV / (denomA + denomB);
	}

	//clamps roughness
	__device__ float D_GGX(float NoH, float roughness)
	{
		roughness = fmaxf(roughness, Constants::MAT_MIN_ROUGHNESS);//needed TODO: switch to specular brdf below this threshold
		float alpha = Sqr(roughness);
		float alpha2 = Sqr(alpha);
		float NoH2 = Sqr(NoH);
		float b = (NoH2 * (alpha2 - 1.0) + 1.0);
		return alpha2 / (Constants::PI * Sqr(b));
	}

	__device__ float fresnelDielectric(float cosTheta, float ior)
	{
		cosTheta = clamp(cosTheta, -1.0f, 1.0f);
		if (cosTheta < 0.0f)
		{
			ior = 1.0f / ior;
			cosTheta = -cosTheta;
		}

		float sin2Theta = (1.0f - cosTheta * cosTheta);
		float sin2Theta_t = sin2Theta / (ior * ior);
		if (sin2Theta_t >= 1.0f) return 1.0f;

		float cosTheta_t = sqrt(1.0f - sin2Theta_t);

		float r_prl = (ior * cosTheta - cosTheta_t) / (ior * cosTheta + cosTheta_t);
		float r_per = (cosTheta - ior * cosTheta_t) / (cosTheta + ior * cosTheta_t);
		return (r_prl * r_prl + r_per * r_per) * 0.5f;
	}

	__device__ float microFacetBRDF(float3 wo, float3 wi, float3 h, float roughness)
	{
		float NoH = clamp(h.z, 0.f, 1.f);
		float NoV = clamp(wo.z, 0.f, 1.f);
		float NoL = clamp(wi.z, 0.f, 1.f);

		//below expects squared roughness
		float D = D_GGX(NoH, roughness);
		float G = G2_Smith(wo, wi, roughness);
		float out = (D * G) / (4 * fmaxf(NoV, 0.0001f) * NoL);
		if (wi.z <= 0.0f || dot(wi, h) <= 0)
		{
			out = 0.0f;
		}
		return out;
	}
	//-----

	__device__ float3 BSDF::fGlossyMicrofacetBRDF(float3 wo, float3 wi, float3 h) const
	{
		float Mss = microFacetBRDF(wo, wi, h, roughness);

		float Fss = fresnelDielectric(dot(wo, h), ior);
		float3 F = make_float3(Fss * Mss);
		return F;
	}

	__device__ float BSDF::pdfGlossyMicrofacetBRDF(float3 wo, float3 wi, float3 h) const
	{
		float VoH = fmaxf(dot(wo, h), 0.0f);
		float NoH = fmaxf(h.z, 0.0f);
		float D = D_GGX(NoH, roughness);
		float pdf = (VoH > 0.f) ? (D * NoH) / (4.0f * VoH) : 0.f;
		return pdf;
	}

	__device__ float3 BSDF::sampleGlossyMicrofacetBRDF_VNDF(float3 wo, float2 u2) const
	{
		float a = Sqr(roughness);
		float a2 = a * a;
		// -- Generate uniform random variables between 0 and 1
		float e0 = u2.x;
		float e1 = u2.y;

		float theta = acosf(sqrtf((1.0f - e0) / ((a2 - 1.0f) * e0 + 1.0f)));
		// Correct GGX sampling of theta using the inverse CDF
		//float theta = atanf(a * sqrtf(e0 / (1.0f - e0)));
		//float theta = acosf(sqrtf(a2 / e0 * (a2 - 1.0f) + 1.0f));//github
		float phi = 2 * Constants::PI * e1;

		float3 wm = sphericalToCartesian(theta, phi);

		return normalize(wm);
	};

	//===========================================================================================================
	//OPAQUE DIELECTRIC BSDF
	//===========================================================================================================

	__device__ BSDFSample BSDF::sampleOpaqueDielectric(float3 wo, float2 u2, float Xi) const
	{
		float path_probability = Xi;

		float glossy_prob = 0.5;

		if (path_probability < glossy_prob)
		{
			//Glossy scatter path----------------------

			float3 h = sampleGlossyMicrofacetBRDF_VNDF(wo, u2);
			float3 wi = reflect(-wo, h);

			float3 f = fGlossyMicrofacetBRDF(wo, wi, h);
			f *= (1.0f / glossy_prob);

			float pdf = pdfGlossyMicrofacetBRDF(wo, wi, h);

			return BSDFSample(BSDFSample::Diffuse | BSDFSample::Reflected, f, wi, pdf);
		}

		//Diffuse scatter path----------------------

		float3 wi = sampleDiffuseBRDF(u2);

		float3 f = albedo_factor * fDiffuseBRDF(wo, wi);
		f *= (1.0f / (1.0f - glossy_prob));

		float pdf = pdfDiffuseBRDF(wo, wi);

		return BSDFSample(BSDFSample::Diffuse | BSDFSample::Reflected, f, wi, pdf);
	}

	__device__ float3 BSDF::fOpaqueDielectric(float3 wo, float3 wi) const
	{
		float3 h = normalize(wo + wi);

		float3 glossy = fGlossyMicrofacetBRDF(wo, wi, h);
		float3 cDiffuse = (1.0f - glossy);

		float3 diffuse = cDiffuse * albedo_factor * fDiffuseBRDF(wo, wi);

		return diffuse + glossy;
	}

	__device__ float BSDF::pdfOpaqueDielectric(float3 wo, float3 wi) const
	{
		float3 h = normalize(wo + wi);
		float pSpec = pdfGlossyMicrofacetBRDF(wo, wi, h);

		float pDiffuse = pdfDiffuseBRDF(wo, wi);

		return (0.5f) * (pSpec + pDiffuse);
	}

	//===========================================================================================================
	//CONDUCTOR BSDF
	//===========================================================================================================
	__device__ float3 fresnelSchlick(float VoH, float3 F0)
	{
		float3 F = F0 + (1.0 - F0) * ::powf(1.0 - VoH, 5.0);
		return clamp(F, 0, 1);
	}

	__device__ BSDFSample BSDF::sampleConductor(float3 wo, float2 u2, float X) const
	{
		float path_probability = X;

		float3 h = sampleGlossyMicrofacetBRDF_VNDF(wo, u2);
		float3 wi = reflect(-wo, h);
		float3 f = fConductor(wo, wi, albedo_factor);
		float pdf = pdfGlossyMicrofacetBRDF(wo, wi, h);

		return BSDFSample(BSDFSample::Diffuse | BSDFSample::Reflected, f, wi, pdf);
	}
	__device__ float3 BSDF::fConductor(float3 wo, float3 wi, float3 albedo) const
	{
		float3 h = normalize(wo + wi);
		float NoV = clamp(wo.z, 0.f, 1.f);
		float NoL = clamp(wi.z, 0.f, 1.f);
		if (NoV == 0 || NoL == 0)
		{
			return make_float3(0);
		}

		float VoH = clamp(dot(wo, h), 0.f, 1.f);
		float3 M = make_float3(0);
		if (NoL > 0 && VoH > 0)
		{
			float3 F = fresnelSchlick(VoH, albedo);
			M = F * microFacetBRDF(wo, wi, h, roughness);
		}
		return M;
	}

	__device__ float BSDF::pdfConductor(float3 wo, float3 wi) const
	{
		float3 h = normalize(wo + wi);
		return pdfGlossyMicrofacetBRDF(wo, wi, h);
	}
}