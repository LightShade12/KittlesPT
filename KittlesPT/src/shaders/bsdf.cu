#include "bsdf.cuh"
#include "samplers.cuh"
#include "../maths/linear_algebra.cuh"

namespace KittlesPT
{
	BSDF::BSDF(const Mat3& tangent_basis,
		float3 albedo,
		float metallicity,
		float roughness,
		float transmission,
		float ior,
		bool is_backface) :
		albedo_factor(albedo),
		metallicity(metallicity),
		roughness(roughness),
		transmission(transmission),
		IOR(ior),
		backface(is_backface)
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
		float prob_trans = (1.0f - metallicity) * transmission;

		BSDFSample bs;

		if (path_probability < prob_metallic)
		{
			bs = sampleConductor(wo, u2, X2.y);
		}
		else if (path_probability < prob_trans)
		{
			bs = sampleTransparentDielectric(wo, u2, X2.y);
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

		float Fss = fresnelDielectric(dot(wo, h), IOR);
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

			return BSDFSample(BSDFSample::Glossy | BSDFSample::Reflected, f, wi, pdf);
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

		return BSDFSample(BSDFSample::Glossy | BSDFSample::Reflected, f, wi, pdf);
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

	//===========================================================================================================
	//TRANSPARENT DIELECTRIC BSDF
	//===========================================================================================================

	__device__ float BSDF::pdfGlossyMicrofacetBTDF(float3 wo, float3 wi, float3 ht, float ior) const
	{
		const float temp = dot(wi, ht) * ior + dot(wo, ht);
		const float dwm_dwi = AbsDot(wi, ht) / (temp * temp);

		const float NoH = fabs(ht.z);
		const float D = D_GGX(NoH, roughness);
		float Fss = fresnelDielectric(AbsDot(wo, ht), ior);
		const float pdf = (D * NoH) * dwm_dwi * (1.0f - Fss);

		return pdf;
	}
	__device__ float3 BSDF::fGlossyMicrofacetBTDF(float3 wo, float3 wi, float3 ht, float ior) const
	{
		const float NoH = fabs(ht.z);
		const float temp = dot(wi, ht) * ior + dot(wo, ht);
		float Fss = fresnelDielectric(AbsDot(wo, ht), ior);
		const float Tss =
			D_GGX(NoH, roughness) * G2_Smith(wo, wi, roughness) *
			fabs(dot(wi, ht) * dot(wo, ht) / (wi.z * wo.z * temp * temp));

		float3 f = (1.0f - Fss) * Tss * albedo_factor;

		return f;
	}

	__device__ BSDFSample BSDF::sampleTransparentDielectric(float3 wo, float2 u2, float X) const
	{
		float path_probability = X;

		float ior = (backface) ? 1.0f / IOR : IOR;

		float3 ht = sampleGlossyMicrofacetBRDF_VNDF(wo, u2);
		float Fss = fresnelDielectric(AbsDot(wo, ht), ior);

		float3 wi;
		bool tir = !refract(wo, ht, ior, wi);

		if ((path_probability < Fss) || tir)
		{
			//Reflection path---------

			wi = reflect(-wo, ht);
			if (!sameHemisphere(wi, wo, make_float3(0, 0, 1)))
			{
				return BSDFSample(BSDFSample::Absorbed, { 0,0,0 }, { 0,0,0 }, 0);
			}

			float Mss = microFacetBRDF(wo, wi, ht, roughness);
			float3 f = make_float3(Fss * Mss);
			float pdf = pdfGlossyMicrofacetBRDF(wo, wi, ht);

			return BSDFSample(BSDFSample::Reflected | BSDFSample::Glossy, f, wi, pdf);
		}
		//Refraction path---------
		if (tir || sameHemisphere(wi, wo, make_float3(0, 0, 1)) || wi.z == 0.0f)
		{
			return BSDFSample(BSDFSample::Absorbed, { 0,0,0 }, { 0,0,0 }, 0);
		}

		//BTDF
		const float temp = dot(wi, ht) * ior + dot(wo, ht);

		const float pdf = pdfGlossyMicrofacetBTDF(wo, wi, ht, ior);

		float3 f = fGlossyMicrofacetBTDF(wo, wi, ht, ior);
		/*refract(wo, make_float3(0, 0, 1), ior, wi);
		f = make_float3(1);
		pdf = 1.0f;*/
		return BSDFSample(BSDFSample::Transmitted | BSDFSample::Glossy, f, wi, pdf);
	}
	__device__ float3 BSDF::fTransparentDielectric(float3 wo, float3 wi) const
	{
		const float cos_theta_o = wo.z, cos_theta_i = wi.z;
		const bool is_reflection = cos_theta_o * cos_theta_i > 0.0f;
		float ior = 1.0f;
		if (!is_reflection)
		{
			ior = (cos_theta_o > 0.0f) ? IOR : 1.0f / IOR;//entry exit determination
		}

		// Calculate microfacet normal
		float3 h = (ior * wi + wo);
		h = normalize((h.z > 0) ? h : -h);

		if (dot(h, wi) * cos_theta_i < 0.0f || dot(h, wo) * cos_theta_o < 0.0f)
		{
			return make_float3(0); // Discard back-facing microsurfaces
		}

		const float Fss = fresnelDielectric(fabs(dot(wo, h)), ior);
		const float T = 1.0f - Fss;

		if (is_reflection)
		{
			// Single-scattering term
			const float Mss = microFacetBRDF(wo, wi, h, roughness);

			return make_float3(Fss * Mss);
		}

		//Refraction---------------

		const float temp = dot(wi, h) * ior + dot(wo, h);
		const float dwm_dwi = fabs(dot(wi, h)) * fabs(dot(wo, h)) / (temp * temp);

		// Single-scattering term
		const float Tss = D_GGX(h.z, roughness) * G2_Smith(wo, wi, roughness) * dwm_dwi /
			(fabs(cos_theta_i * cos_theta_o));

		return T * albedo_factor * Tss;
	}
	__device__ float BSDF::pdfTransparentDielectric(float3 wo, float3 wi) const
	{
		const float cos_theta_o = wo.z, cos_theta_i = wi.z;
		const bool is_reflection = cos_theta_o * cos_theta_i > 0.0f;
		float ior = 1.0f;
		if (!is_reflection)
		{
			ior = (cos_theta_o > 0.0f) ? IOR : 1.0f / IOR;//entry exit determination
		}

		// Calculate microfacet normal
		float3 h = (ior * wi + wo);
		h = normalize((h.z > 0) ? h : -h);

		if (dot(h, wi) * cos_theta_i < 0.0f || dot(h, wo) * cos_theta_o < 0.0f)
		{
			return 0.0f; // Discard back-facing microsurfaces
		}

		const float Fss = fresnelDielectric(fabs(dot(wo, h)), ior);
		const float T = 1.0f - Fss;

		float pdf;

		if (is_reflection)
		{
			float VoH = fmaxf(dot(wo, h), 0.0f);
			float NoH = fmaxf(h.z, 0.0f);
			float D = D_GGX(NoH, roughness);
			pdf = (VoH > 0.f) ? (D * NoH) / (4.0f * VoH) * Fss : 0.f;
		}
		else
		{
			const float temp = dot(wi, h) + dot(wo, h) / ior;
			const float dwm_dwi = fabs(dot(wo, h)) / (temp * temp);
			float NoH = fmaxf(h.z, 0.0f);
			float D = D_GGX(NoH, roughness);
			pdf = D * NoH * dwm_dwi * T;
		}

		return pdf;
	}
}