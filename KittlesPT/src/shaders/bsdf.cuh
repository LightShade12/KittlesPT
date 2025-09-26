#pragma once
#include "samplers.cuh"
#include "maths/linear_algebra.cuh"
#include "maths/constants.cuh"
#include "color.cuh"

namespace KittlesPT
{
	//__constant__ constexpr float GGX_ROUGHNESS_EPSILON = 0.045f;

	inline __device__ float computeF0(float ior) {
		return Sqr((ior - 1.0f) / (ior + 1.0f));
	}

	/// <summary>
	/// Trowbridge-Reitz Normal Distribution Model
	/// </summary>
	class GGX
	{
	public:
		__device__ GGX() = default;
		__device__ GGX(float roughness) :
			m_alpha(Sqr(roughness))
		{}

		__device__ float D(const float3& wm) const
		{
			//roughness = fmaxf(roughness, GGX_ROUGHNESS_EPSILON);//needed TODO: switch to delta specular brdf below this threshold
			float alpha = m_alpha;
			if (isEffectivelySmooth()) {
				alpha = fmaxf(alpha, 1.0e-3f);//TODO: switch to specular
			}
			float alpha2 = Sqr(alpha);
			float N_dot_H = Sqr(saturate(wm.z));
			float b = (N_dot_H * (alpha2 - 1.0) + 1.0);
			return alpha2 / (Constants::PI * Sqr(b));
		}

		//TODO: use GGX G1
		//Heitz 2014
		inline __device__ float G1(float theta) {
			return 2.0f / (1.0f + sqrtf(1.0f + Sqr(m_alpha * tanf(theta))));
		}

		__device__ float G(const float3& wo, const float3& wi) const
		{
			float dotNV = fmaxf(wo.z, 0.0f);
			float dotNL = fabsf(wi.z);

			float sqrtTermNV = sqrtf(Sqr(m_alpha) + (1.0f - Sqr(m_alpha)) * Sqr(dotNV));
			float sqrtTermNL = sqrtf(Sqr(m_alpha) + (1.0f - Sqr(m_alpha)) * Sqr(dotNL));

			float G1_NV = (2.0f * dotNV) / (dotNV + sqrtTermNV);
			float G1_NL = (2.0f * dotNL) / (dotNL + sqrtTermNL);

			return G1_NV * G1_NL;
		}

		__device__ float3 sampleWm(float3 w, float2 u2) const
		{
			float a2 = Sqr(m_alpha);

			float e0 = u2.x;
			float e1 = u2.y;

			float theta = acosf(sqrtf((1.0f - e0) / ((a2 - 1.0f) * e0 + 1.0f)));
			//float theta = atanf(a * sqrtf(e0 / (1.0f - e0)));//chatgpt
			//float theta = acosf(sqrtf(a2 / e0 * (a2 - 1.0f) + 1.0f));//Q2RTX github
			float phi = 2.0f * Constants::PI * e1;

			float3 wm = sphericalToCartesian(theta, phi);

			return normalize(wm);
		};

		__device__ bool isEffectivelySmooth() const {
			return m_alpha < 0.001f;
		}

		__device__ void regularize() {
			if (m_alpha < 0.3f) {
				m_alpha = clamp(2.0f * m_alpha, 0.1f, 0.3f);
			}
		}

	private:
		float m_alpha = 0.25f;
	};

	struct BSDFSample
	{
		BSDFSample() = default;

		__device__ BSDFSample(int scatter, RGBSpectrum f, float3 wi, float pdf) :
			scatter(scatter), f(f), wi(wi), pdf(pdf) {};

		__device__ bool scatterTypeIs(int flag) const
		{
			return scatter & flag;
		}

		enum Scatter
		{
			Absorbed = 0,
			Emitted = 1 << 0,
			Reflected = 1 << 1,
			Transmitted = 1 << 2,
			Diffuse = 1 << 3,
			Glossy = 1 << 4,
			Specular = 1 << 5
		};

		int scatter = Absorbed;
		RGBSpectrum f{ 0.0f,0.0f,0.0f };
		float3 wi{ 0.0f, 0.0f, 0.0f };
		float pdf = 0.0f;
	};

	//===================================================================================================================================================

	//FrDielectric; costheta = wo dot wm
	inline __device__ float fresnelDielectric(float cosTheta, float ior)
	{
		cosTheta = clamp(cosTheta, -1.0f, 1.0f);
		if (cosTheta < 0.0f) {
			ior = 1.0f / ior;
			cosTheta = -cosTheta;
		}

		float sin2Theta = (1.0f - cosTheta * cosTheta);
		float sin2Theta_t = sin2Theta / (ior * ior);
		if (sin2Theta_t >= 1.0f) {
			return 1.0f;
		}

		float cosTheta_t = sqrtf(1.0f - sin2Theta_t);

		float r_prl = (ior * cosTheta - cosTheta_t) / (ior * cosTheta + cosTheta_t);
		float r_per = (cosTheta - ior * cosTheta_t) / (cosTheta + ior * cosTheta_t);
		return (r_prl * r_prl + r_per * r_per) * 0.5f;
	}

	//FrSchlick approximation
	inline __device__ RGBSpectrum fresnelSchlick(float cos_theta, RGBSpectrum F0)
	{
		RGBSpectrum F = F0 + (1.0f - F0) * pow5(1.0f - cos_theta);
		return clamp(F, RGBSpectrum(0.0f), RGBSpectrum(1.0f));
	}

	//Cook-Torrance Microfacet BRDF equation
	inline __device__ float microfacetBRDF(float3 wo, float3 wi, float3 wm, const GGX& ggx)
	{
		float NdotV = saturate(wo.z);
		float NdotL = saturate(wi.z);

		//below expects squared roughness(alpha)
		float D = ggx.D(wm);//The NDF
		float G = ggx.G(wo, wi);//Geometry term(Shadowing-Masking)
		float f = (D * G) / (4.0f * fmaxf(NdotV, 1.0e-3f) * NdotL);
		if (wi.z <= 0.0f || dot(wi, wm) <= 0.0f) {
			f = 0.0f;
		}
		return f;
	}

	class UnifiedBSDF
	{
	public:

		__device__ UnifiedBSDF(float3 t, float3 b, float3 n) :m_tangent_basis(Mat3(t, b, n)) {};

		__device__ UnifiedBSDF(const Mat3& tangent_basis,
			RGBSpectrum albedo,
			float metallicity,
			float roughness,
			float transmission,
			float ior,
			bool is_backface) :
			m_tangent_basis(tangent_basis),
			m_albedo(albedo),
			m_metallicity(metallicity),
			m_roughness(roughness),
			m_transmission(transmission),
			m_IOR(ior),
			m_is_backface(is_backface),
			m_GGX_mf(roughness)
		{}

		//------------------------------------

		__device__ RGBSpectrum f(float3 w_wo, float3 w_wi) const
		{
			Mat3 inv_basis = m_tangent_basis.transpose();
			float3 wo = inv_basis * w_wo;
			float3 wi = inv_basis * w_wi;

			float w_metallic = (m_metallicity);
			float w_transmissive_dielectric = (1.0f - w_metallic) * m_transmission;
			float w_opaque_dielectric = (1.0f - w_metallic) * (1.0f - m_transmission);

			RGBSpectrum F(0.0f);

			if (w_metallic > 0.0f) {
				F += (fConductor(wo, wi) * w_metallic);
			}
			if (w_transmissive_dielectric > 0.0f) {
				F += (fTransparentDielectric(wo, wi) * w_transmissive_dielectric);
			}
			if (w_opaque_dielectric > 0.0f) {
				F += (fOpaqueDielectric(wo, wi) * w_opaque_dielectric);
			}

			//return m_albedo * fDiffuseBRDF(wo, wi);
			//return fConductor(wo, wi);
			return F;
		}

		__device__ float pdf(float3 w_wo, float3 w_wi) const
		{
			Mat3 inv_basis = m_tangent_basis.transpose();
			float3 wi = inv_basis * w_wi;
			float3 wo = inv_basis * w_wo;

			float w_metallic = m_metallicity;
			float w_transmissive_dielectric = (1.0f - w_metallic) * m_transmission;
			float w_opaque_dielectric = (1.0f - w_metallic) * (1.0f - m_transmission);

			float pdf = 0.0f;
			if (w_metallic > 0.0f) {
				pdf += w_metallic * pdfConductor(wo, wi);
			}
			if (w_transmissive_dielectric > 0.0f) {
				pdf += w_transmissive_dielectric * pdfTransparentDielectric(wo, wi);
			}
			if (w_opaque_dielectric > 0.0f) {
				pdf += w_opaque_dielectric * pdfOpaqueDielectric(wo, wi);
			}
			//return pdfConductor(wo, wi);
			//return pdfTransparentDielectric(wo, wi);
			return pdf;
		}

		__device__ BSDFSample sampleF(float3 w_wo, float2 u2, float2 X2) const
		{
			float path_probability = X2.x;
			float3 wo = m_tangent_basis.transpose() * w_wo;

			float prob_metallic = m_metallicity;
			float prob_trans = (1.0f - prob_metallic) * m_transmission;

			BSDFSample bs;

			if (path_probability < prob_metallic) {
				bs = sampleConductor(wo, u2, X2.y);
			}
			else if (path_probability < prob_trans) {
				bs = sampleTransparentDielectric(wo, u2, X2.y);
			}
			else {
				bs = sampleOpaqueDielectric(wo, u2, X2.y);
			}

			//bs = sampleConductor(wo, u2, X2.y);
			//bs = sampleTransparentDielectric(wo, u2, X2.y);

			bs.wi = m_tangent_basis * bs.wi;
			return bs;
		}

		//TODO: add threshold constant
		__device__ void regularize() {
			if (m_roughness < 0.3f) {
				m_roughness = clamp(2.0f * m_roughness, 0.1f, 0.3f);
			}
			m_GGX_mf.regularize();
		}

		__device__ bool isNonSpecular() {
			return !m_GGX_mf.isEffectivelySmooth();
		}

		__device__ bool operator! () {
			return m_is_medium;
		}

		__device__ void demodulate() {
			m_albedo = RGBSpectrum(1.0f);
		}

		//basically low quality Rho estimate
		__device__ RGBSpectrum getAlbedo() {
			return m_albedo;
		}

	private:
		//BSDFs========================================================
		//Opaque Dielectric BSDF--------------
		__device__ BSDFSample sampleOpaqueDielectric(float3 wo, float2 u2, float Xi) const
		{
			float path_probability = Xi;

			constexpr float glossy_prob = 0.5f;//TODO: make this adaptive

			if (path_probability < glossy_prob)
			{
				//Glossy scatter path----------------------

				float3 wm = m_GGX_mf.sampleWm(wo, u2);
				float3 wi = reflect(-wo, wm);

				RGBSpectrum f = fGlossyMicrofacetBRDF(wo, wi, wm);
				f *= (1.0f / glossy_prob);

				float pdf = pdfGlossyMicrofacetBRDF(wo, wi, wm);

				return BSDFSample(BSDFSample::Glossy | BSDFSample::Reflected, f, wi, pdf);
			}

			//Diffuse scatter path----------------------

			float3 wi = sampleDiffuseBRDF(u2);

			RGBSpectrum f = m_albedo * fDiffuseBRDF(wo, wi);
			f *= (1.0f / (1.0f - glossy_prob));

			float pdf = pdfDiffuseBRDF(wo, wi);

			return BSDFSample(BSDFSample::Diffuse | BSDFSample::Reflected, f, wi, pdf);
		}

		__device__ RGBSpectrum fOpaqueDielectric(float3 wo, float3 wi) const
		{
			float3 wm = normalize(wo + wi);

			RGBSpectrum glossy = fGlossyMicrofacetBRDF(wo, wi, wm);
			RGBSpectrum w_diffuse = RGBSpectrum(1.0f - glossy);

			RGBSpectrum diffuse = w_diffuse * (m_albedo * fDiffuseBRDF(wo, wi));

			return diffuse + glossy;
		}

		__device__ float pdfOpaqueDielectric(float3 wo, float3 wi) const
		{
			float3 wm = normalize(wo + wi);

			float p_specular = pdfGlossyMicrofacetBRDF(wo, wi, wm);

			float p_diffuse = pdfDiffuseBRDF(wo, wi);

			return (0.5f) * (p_specular + p_diffuse);
		}

		//Conductor BSDF---------
		__device__ BSDFSample sampleConductor(float3 wo, float2 u2, float X) const
		{
			//float path_probability = X;

			float3 wm = m_GGX_mf.sampleWm(wo, u2);
			float3 wi = reflect(-wo, wm);
			RGBSpectrum f = fConductor(wo, wi);
			float pdf = pdfGlossyMicrofacetBRDF(wo, wi, wm);

			return BSDFSample(BSDFSample::Glossy | BSDFSample::Reflected, f, wi, pdf);
		}

		__device__ RGBSpectrum fConductor(float3 wo, float3 wi) const
		{
			float3 wm = normalize(wo + wi);
			float NoV = clamp(wo.z, 0.0f, 1.0f);
			float NoL = clamp(wi.z, 0.0f, 1.0f);
			if (NoV == 0.0f || NoL == 0.0f)
			{
				return RGBSpectrum(0.0f);
			}

			float VoH = clamp(dot(wo, wm), 0.0f, 1.0f);
			RGBSpectrum M{ 0.0f,0.0f,0.0f };
			if (NoL > 0.0f && VoH > 0.0f)
			{
				RGBSpectrum F = RGBSpectrum(fresnelSchlick(VoH, m_albedo));
				M = F * microfacetBRDF(wo, wi, wm, m_GGX_mf);
			}
			return M;
		}

		__device__ float pdfConductor(float3 wo, float3 wi) const
		{
			float3 wm = normalize(wo + wi);
			return pdfGlossyMicrofacetBRDF(wo, wi, wm);
		}

		//Transparent Dielectric BSDF--------------
		__device__ BSDFSample sampleTransparentDielectric(float3 wo, float2 u2, float X) const
		{
			float path_probability = X;

			const bool inside_volume = m_is_backface;

			float ior = (inside_volume) ? (1.0f / m_IOR) : m_IOR;

			float3 ht = m_GGX_mf.sampleWm(wo, u2);

			float3 wi;
			bool tir = !refract(wo, ht, ior, wi);
			float glossy_prob = 0.5f;

			if ((path_probability < glossy_prob && !inside_volume) || (tir && inside_volume))
			{
				//Reflection path---------

				wi = reflect(-wo, ht);

				//if somehow reflection is not on same the hemisphere
				if (!sameHemisphere(wi, wo, make_float3(0, 0, 1)))
				{
					return BSDFSample(BSDFSample::Absorbed, { 0,0,0 }, { 0,0,0 }, 0);
				}

				float Mss = microfacetBRDF(wo, wi, ht, m_GGX_mf);
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

		__device__ RGBSpectrum fTransparentDielectric(float3 wo, float3 wi) const
		{
			const float cos_theta_o = wo.z, cos_theta_i = wi.z;
			const bool is_reflection = (cos_theta_o * cos_theta_i) > 0.0f;//same as sameHemisphere()
			float ior = 1.0f;

			if (!is_reflection)
			{
				ior = (!m_is_backface) ? m_IOR : (1.0f / m_IOR);//entry exit determination
			}

			// Calculate microfacet normal
			float3 wm = (ior * wi + wo);
			wm = normalize((wm.z > 0) ? wm : -wm);

			if ((dot(wm, wi) * cos_theta_i < 0.0f) || (dot(wm, wo) * cos_theta_o < 0.0f))
			{
				return RGBSpectrum(0); // Discard back-facing microsurfaces
			}

			const float Fss = fresnelDielectric(fabs(dot(wo, wm)), m_IOR);
			const float T = 1.0f - Fss;

			if (is_reflection)
			{
				// Single-scattering term
				const float Mss = microfacetBRDF(wo, wi, wm, m_GGX_mf);

				return RGBSpectrum(Fss * Mss);
			}

			//Refraction---------------

			const float temp = dot(wi, wm) * ior + dot(wo, wm);
			const float dwm_dwi = fabs(dot(wi, wm)) * fabs(dot(wo, wm)) / (temp * temp);

			// Single-scattering term
			const float Tss = m_GGX_mf.D(wm) * m_GGX_mf.G(wo, wi) * dwm_dwi /
				(fabs(cos_theta_i * cos_theta_o));

			return RGBSpectrum(T * m_albedo * Tss);
		}

		__device__ float pdfTransparentDielectric(float3 wo, float3 wi) const
		{
			const float cos_theta_o = wo.z, cos_theta_i = wi.z;
			const bool is_reflection = cos_theta_o * cos_theta_i > 0.0f;
			float ior = 1.0f;
			if (!is_reflection)
			{
				ior = (!m_is_backface) ? m_IOR : 1.0f / m_IOR;//entry exit determination
			}

			// Calculate microfacet normal
			float3 wm = (ior * wi + wo);
			wm = normalize((wm.z > 0) ? wm : -wm);

			if (dot(wm, wi) * cos_theta_i < 0.0f || dot(wm, wo) * cos_theta_o < 0.0f)
			{
				return 0.0f; // Discard back-facing microsurfaces
			}

			const float Fss = fresnelDielectric(fabs(dot(wo, wm)), m_IOR);
			const float T = 1.0f - Fss;

			float pdf;

			if (is_reflection)
			{
				pdf = pdfGlossyMicrofacetBRDF(wo, wi, wm) * Fss;
			}
			else
			{
				const float temp = dot(wi, wm) + dot(wo, wm) / ior;
				const float dwm_dwi = fabs(dot(wo, wm)) / (temp * temp);
				float NoH = fmaxf(wm.z, 0.0f);
				float D = m_GGX_mf.D(wm);
				pdf = D * NoH * dwm_dwi * T;
			}
			//if (!backface)
			//{
			//	pdf *= 0.5;//factor glossy prob
			//}

			return pdf;
		}

		//BxDFs========================================================

		//Diffuse BRDF
		__device__ float3 sampleDiffuseBRDF(float2 u2) const
		{
			return sampleCosineWeightedHemisphere(u2);
		}
		__device__ float fDiffuseBRDF(float3 wo, float3 wi) const
		{
			float scalar_switch = (wo.z * wi.z > 0) ? 1.0f : 0.0f;//same as sameHemisphere()
			float out = scalar_switch * Constants::INV_PI;
			return out;
		}
		__device__ float pdfDiffuseBRDF(float3 wo, float3 wi) const
		{
			return wi.z * Constants::INV_PI;
		}

		//Specular BRDF
		__device__ float3 sampleSpecularBRDF(float3 wo)
		{
			return reflect(-wo, make_float3(0, 0, 1));
		}
		__device__ float fSpecularBRDF(float3 wo, float3 wi) {
			return 0.0f;
		}
		__device__ float pdfSpecularBRDF(float3 wo, float3 wi) {
			return 0.0f;
		}

		//Microfacet glossy BRDF
		__device__ RGBSpectrum fGlossyMicrofacetBRDF(float3 wo, float3 wi, float3 wm) const
		{
			float Mss = microfacetBRDF(wo, wi, wm, m_GGX_mf);
			float Fss = fresnelDielectric(dot(wo, wm), m_IOR);

			return RGBSpectrum(Fss * Mss);
		}
		__device__ float pdfGlossyMicrofacetBRDF(float3 wo, float3 wi, float3 wm) const
		{
			float VoH = fmaxf(dot(wo, wm), 0.0f);
			float NoH = fmaxf(wm.z, 0.0f);
			float D = m_GGX_mf.D(wm);
			float pdf = (VoH > 0.0f) ? (D * NoH) / (4.0f * VoH) : 0.0f;
			pdf = (D * NoH) / (4.0f * VoH);
			return pdf;
		}

		//Microfacet glossy BTDF
		__device__ float pdfGlossyMicrofacetBTDF(float3 wo, float3 wi, float3 ht, float ior) const
		{
			const float temp = dot(wi, ht) * ior + dot(wo, ht);
			const float dwm_dwi = AbsDot(wi, ht) / (temp * temp);

			const float NoH = fabsf(ht.z);
			const float D = m_GGX_mf.D(ht);
			float Fss = fresnelDielectric(AbsDot(wo, ht), ior);
			const float pdf = (D * NoH) * dwm_dwi * (1.0f - Fss);

			return pdf;
		}
		__device__ RGBSpectrum fGlossyMicrofacetBTDF(float3 wo, float3 wi, float3 ht, float ior) const
		{
			const float NoH = fabsf(ht.z);
			const float temp = dot(wi, ht) * ior + dot(wo, ht);
			float Fss = fresnelDielectric(AbsDot(wo, ht), ior);
			const float Tss =
				m_GGX_mf.D(ht) * m_GGX_mf.G(wo, wi) *
				fabs(dot(wi, ht) * dot(wo, ht) / (wi.z * wo.z * temp * temp));

			return m_albedo * (1.0f - Fss) * Tss;
		}

		//---------------------------------------------------------------------------------------------------

	private:
		GGX m_GGX_mf;
		Mat3 m_tangent_basis;
		RGBSpectrum m_albedo{ 1.0f,0.0f,1.0f };//reflectance spectrum
		float m_roughness = 0.5f;//alpha_x alpha_y TODO: anisotropy
		float m_metallicity = 0.0f;
		float m_transmission = 0.0f;
		float m_IOR = 1.45f;//eta
		bool m_is_backface = false;
		bool m_is_medium = false;
	};
}/*KittlesPT*/