#include "bsdf.cuh"
#include "samplers.cuh"
#include "maths/linear_algebra.cuh"

namespace KittlesPT
{
	__device__ BSDF::BSDF(const Mat3& tangent_basis,
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

	__device__ RGBSpectrum BSDF::f(float3 w_wo, float3 w_wi) const
	{
		float3 wo = tangent_matrix.inverse() * w_wo;
		float3 wi = tangent_matrix.inverse() * w_wi;

		float w_metallic = metallicity;
		float w_transmissive_dielectric = (1.0f - w_metallic) * transmission;
		float w_opaque_dielectric = (1.0f - w_metallic) * (1.0f - transmission);

		RGBSpectrum F(0);

		if (w_metallic > 0.0f) {
			F += (fConductor(wo, wi) * w_metallic);
		}
		if (w_transmissive_dielectric > 0.0f) {
			F += (fTransparentDielectric(wo, wi) * w_transmissive_dielectric);
		}
		if (w_opaque_dielectric > 0.0f) {
			F += (fOpaqueDielectric(wo, wi) * w_opaque_dielectric);
		}

		//return albedo_factor * fDiffuseBRDF(wo, wi);
		//return fTransparentDielectric(wo, wi);
		return F;
	}

	__device__ float BSDF::pdf(float3 w_wo, float3 w_wi) const
	{
		float3 wi = tangent_matrix.inverse() * w_wi;
		float3 wo = tangent_matrix.inverse() * w_wo;

		float w_metallic = metallicity;
		float w_transmissive_dielectric = (1.0f - w_metallic) * transmission;
		float w_opaque_dielectric = (1.0f - w_metallic) * (1.0f - transmission);

		float pdf = 0;
		if (w_metallic > 0.f) {
			pdf += w_metallic * pdfConductor(wo, wi);
		}
		if (w_transmissive_dielectric > 0.f) {
			pdf += w_transmissive_dielectric * pdfTransparentDielectric(wo, wi);
		}
		if (w_opaque_dielectric > 0.f) {
			pdf += w_opaque_dielectric * pdfOpaqueDielectric(wo, wi);
		}
		//return pdfDiffuseBRDF(wo, wi);
		//return pdfTransparentDielectric(wo, wi);
		return pdf;
	}

	__device__ BSDFSample BSDF::sampleBSDF(float3 w_wo, float2 u2, float2 X2) const
	{
		float path_probability = X2.x;
		float3 wo = tangent_matrix.inverse() * w_wo;

		float prob_metallic = metallicity;
		float prob_trans = (1.0f - prob_metallic) * transmission;

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

		//bs = sampleOpaqueDielectric(wo, u2, X2.y);
		//bs = sampleTransparentDielectric(wo, u2, X2.y);

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
		float scalar_switch = (wo.z * wi.z > 0) ? 1 : 0;//same as sameHemisphere()
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

	//Components-----------------------------------------------------------------

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

	//GGX NDF; clamps roughness
	__device__ float D_GGX(float NoH, float roughness)
	{
		roughness = fmaxf(roughness, Constants::MAT_MIN_ROUGHNESS);//needed TODO: switch to specular brdf below this threshold
		float alpha = Sqr(roughness);
		float alpha2 = Sqr(alpha);
		float NoH2 = Sqr(NoH);
		float b = (NoH2 * (alpha2 - 1.0) + 1.0);
		return alpha2 / (Constants::PI * Sqr(b));
	}

	//FrDielectric
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

	//Cook-Torrance microfacet brdf equation
	__device__ float microFacetBRDF(float3 wo, float3 wi, float3 h, float roughness)
	{
		float NoH = clamp(h.z, 0.f, 1.f);
		float NoV = clamp(wo.z, 0.f, 1.f);
		float NoL = clamp(wi.z, 0.f, 1.f);

		//below expects squared roughness(alpha)
		float D = D_GGX(NoH, roughness);
		float G = G2_Smith(wo, wi, roughness);
		float out = (D * G) / (4 * fmaxf(NoV, 0.0001f) * NoL);
		if (wi.z <= 0.0f || dot(wi, h) <= 0)
		{
			out = 0.0f;
		}
		return out;
	}

	//BRDF--------------------------------------------------------

	__device__ RGBSpectrum BSDF::fGlossyMicrofacetBRDF(float3 wo, float3 wi, float3 h) const
	{
		float Mss = microFacetBRDF(wo, wi, h, roughness);

		float Fss = fresnelDielectric(dot(wo, h), IOR);
		RGBSpectrum F = RGBSpectrum(Fss * Mss);
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

		float e0 = u2.x;
		float e1 = u2.y;

		float theta = acosf(sqrtf((1.0f - e0) / ((a2 - 1.0f) * e0 + 1.0f)));
		//float theta = atanf(a * sqrtf(e0 / (1.0f - e0)));//chatgpt
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

		float glossy_prob = 0.5f;

		if (path_probability < glossy_prob)
		{
			//Glossy scatter path----------------------

			float3 h = sampleGlossyMicrofacetBRDF_VNDF(wo, u2);
			float3 wi = reflect(-wo, h);

			RGBSpectrum f = fGlossyMicrofacetBRDF(wo, wi, h);
			f *= (1.0f / glossy_prob);

			float pdf = pdfGlossyMicrofacetBRDF(wo, wi, h);

			return BSDFSample(BSDFSample::Glossy | BSDFSample::Reflected, f, wi, pdf);
		}

		//Diffuse scatter path----------------------

		float3 wi = sampleDiffuseBRDF(u2);

		RGBSpectrum f = albedo_factor * fDiffuseBRDF(wo, wi);
		f *= (1.0f / (1.0f - glossy_prob));

		float pdf = pdfDiffuseBRDF(wo, wi);

		return BSDFSample(BSDFSample::Diffuse | BSDFSample::Reflected, f, wi, pdf);
	}

	__device__ RGBSpectrum BSDF::fOpaqueDielectric(float3 wo, float3 wi) const
	{
		float3 h = normalize(wo + wi);

		RGBSpectrum glossy = fGlossyMicrofacetBRDF(wo, wi, h);
		RGBSpectrum cDiffuse = RGBSpectrum(1.0f - glossy);

		RGBSpectrum diffuse = cDiffuse * (albedo_factor * fDiffuseBRDF(wo, wi));

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

	//Components-----------------------------------------------------

	//FrSchlick approximation
	__device__ float3 fresnelSchlick(float VoH, float3 F0)
	{
		float3 F = F0 + (1.0 - F0) * ::powf(1.0 - VoH, 5.0);
		return clamp(F, 0, 1);
	}

	//BSDF----------------------------------------------------------

	__device__ BSDFSample BSDF::sampleConductor(float3 wo, float2 u2, float X) const
	{
		float path_probability = X;//TODO:unused

		float3 h = sampleGlossyMicrofacetBRDF_VNDF(wo, u2);
		float3 wi = reflect(-wo, h);
		RGBSpectrum f = fConductor(wo, wi);
		float pdf = pdfGlossyMicrofacetBRDF(wo, wi, h);

		return BSDFSample(BSDFSample::Glossy | BSDFSample::Reflected, f, wi, pdf);
	}

	__device__ RGBSpectrum BSDF::fConductor(float3 wo, float3 wi) const
	{
		float3 h = normalize(wo + wi);
		float NoV = clamp(wo.z, 0.f, 1.f);
		float NoL = clamp(wi.z, 0.f, 1.f);
		if (NoV == 0 || NoL == 0)
		{
			return RGBSpectrum(0);
		}

		float VoH = clamp(dot(wo, h), 0.f, 1.f);
		RGBSpectrum M(0);
		if (NoL > 0 && VoH > 0)
		{
			RGBSpectrum F = RGBSpectrum(fresnelSchlick(VoH, albedo_factor.toFloat3()));
			M = F * microFacetBRDF(wo, wi, h, roughness);
		}
		return M;
	}

	__device__ float BSDF::pdfConductor(float3 wo, float3 wi) const
	{
		float3 h = normalize(wo + wi);
		return pdfGlossyMicrofacetBRDF(wo, wi, h);
	}

	//--------------------------------------------------------------------------------------------------------------

	//===========================================================================================================
	//GLOSSY MICROFACET BTDF
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

	__device__ RGBSpectrum BSDF::fGlossyMicrofacetBTDF(float3 wo, float3 wi, float3 ht, float ior) const
	{
		const float NoH = fabs(ht.z);
		const float temp = dot(wi, ht) * ior + dot(wo, ht);
		float Fss = fresnelDielectric(AbsDot(wo, ht), ior);
		const float Tss =
			D_GGX(NoH, roughness) * G2_Smith(wo, wi, roughness) *
			fabs(dot(wi, ht) * dot(wo, ht) / (wi.z * wo.z * temp * temp));

		RGBSpectrum f = albedo_factor * (1.0f - Fss) * Tss;

		return f;
	}

	//===========================================================================================================
	//TRANSPARENT DIELECTRIC BSDF
	//===========================================================================================================

	__device__ BSDFSample BSDF::sampleTransparentDielectric(float3 wo, float2 u2, float X) const
	{
		float path_probability = X;

		const bool inside_volume = backface;

		float ior = (inside_volume) ? (1.0f / IOR) : IOR;

		float3 ht = sampleGlossyMicrofacetBRDF_VNDF(wo, u2);

		float3 wi;
		bool tir = !refract(wo, ht, ior, wi);
		float glossy_prob = 0.5;

		if ((path_probability < glossy_prob && !inside_volume) || (tir && inside_volume))
		{
			//Reflection path---------

			wi = reflect(-wo, ht);

			//if somehow reflection is not on same the hemisphere
			if (!sameHemisphere(wi, wo, make_float3(0, 0, 1)))
			{
				return BSDFSample(BSDFSample::Absorbed, { 0,0,0 }, { 0,0,0 }, 0);
			}

			float Mss = microFacetBRDF(wo, wi, ht, roughness);
			float Fss = fresnelDielectric(dot(wo, ht), ior);
			RGBSpectrum f = RGBSpectrum(Fss * Mss);

			//factor glossy prob if not due to TIR
			if (!tir) {
				f *= (1.0f / glossy_prob);
			}

			float pdf = pdfGlossyMicrofacetBRDF(wo, wi, ht);

			return BSDFSample(BSDFSample::Reflected | BSDFSample::Glossy, f, wi, pdf);
		}

		//Refraction path---------

		//if somehow refraction is on same hemisphere or tir case slipped or degenerate wi
		if (tir || sameHemisphere(wi, wo, make_float3(0, 0, 1)) || wi.z == 0.0f)
		{
			return BSDFSample(BSDFSample::Absorbed, { 0,0,0 }, { 0,0,0 }, 0);
		}

		//BTDF
		const float pdf = pdfGlossyMicrofacetBTDF(wo, wi, ht, ior);
		RGBSpectrum  f = fGlossyMicrofacetBTDF(wo, wi, ht, ior);
		if (!inside_volume) {
			f *= (1.0f / (1.0f - glossy_prob));//factor glossy_prob if entry refraction
		}

		return BSDFSample(BSDFSample::Transmitted | BSDFSample::Glossy, f, wi, pdf);
	}

	__device__ RGBSpectrum BSDF::fTransparentDielectric(float3 wo, float3 wi) const
	{
		const float cos_theta_o = wo.z, cos_theta_i = wi.z;
		const bool is_reflection = (cos_theta_o * cos_theta_i) > 0.0f;//same as sameHemisphere()
		float ior = 1.0f;

		if (!is_reflection)
		{
			ior = (!backface) ? IOR : (1.0f / IOR);//entry exit determination
		}

		// Calculate microfacet normal
		float3 h = (ior * wi + wo);
		h = normalize((h.z > 0) ? h : -h);

		if ((dot(h, wi) * cos_theta_i < 0.0f) || (dot(h, wo) * cos_theta_o < 0.0f))
		{
			return RGBSpectrum(0); // Discard back-facing microsurfaces
		}

		const float Fss = fresnelDielectric(fabs(dot(wo, h)), IOR);
		const float T = 1.0f - Fss;

		if (is_reflection)
		{
			// Single-scattering term
			const float Mss = microFacetBRDF(wo, wi, h, roughness);

			return RGBSpectrum(Fss * Mss);
		}

		//Refraction---------------

		const float temp = dot(wi, h) * ior + dot(wo, h);
		const float dwm_dwi = fabs(dot(wi, h)) * fabs(dot(wo, h)) / (temp * temp);

		// Single-scattering term
		const float Tss = D_GGX(h.z, roughness) * G2_Smith(wo, wi, roughness) * dwm_dwi /
			(fabs(cos_theta_i * cos_theta_o));

		return RGBSpectrum(T * albedo_factor * Tss);
	}

	__device__ float BSDF::pdfTransparentDielectric(float3 wo, float3 wi) const
	{
		const float cos_theta_o = wo.z, cos_theta_i = wi.z;
		const bool is_reflection = cos_theta_o * cos_theta_i > 0.0f;
		float ior = 1.0f;
		if (!is_reflection)
		{
			ior = (!backface) ? IOR : 1.0f / IOR;//entry exit determination
		}

		// Calculate microfacet normal
		float3 h = (ior * wi + wo);
		h = normalize((h.z > 0) ? h : -h);

		if (dot(h, wi) * cos_theta_i < 0.0f || dot(h, wo) * cos_theta_o < 0.0f)
		{
			return 0.0f; // Discard back-facing microsurfaces
		}

		const float Fss = fresnelDielectric(fabs(dot(wo, h)), IOR);
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
		//if (!backface) 
		//{
		//	pdf *= 0.5;//factor glossy prob
		//}

		return pdf;
	}
}