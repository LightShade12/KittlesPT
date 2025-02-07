#pragma once
#include "maths/matrix_maths.cuh"
#include "maths/constants.cuh"
#include "color.cuh"

namespace KittlesPT
{
	/*
	* TODO: GGX distrib class
	* -More flags utility
	*/

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
			Emitted = 1,
			Reflected = 2,
			Transmitted = 4,
			Diffuse = 8,
			Glossy = 16,
			Specular = 32
		};

		int scatter = Absorbed;
		RGBSpectrum f{ 0.0f,0.0f,0.0f };
		float3 wi{ 0.0f, 0.0f, 0.0f };
		float pdf = 0.0f;
	};

	//===================================================================================================================================================

	class BSDF
	{
	public:

		__device__ BSDF(float3 t, float3 b, float3 n) :m_tangent_basis(Mat3(t, b, n)) {};

		__device__ BSDF(const Mat3& tangent_basis,
			RGBSpectrum albedo,
			float metallicity,
			float roughness,
			float transmission,
			float ior,
			bool is_backface);

		//------------------------------------

		__device__ RGBSpectrum f(float3 r_wo, float3 r_wi) const;

		__device__ float pdf(float3 r_wo, float3 r_wi) const;

		__device__ BSDFSample sampleF(float3 w_wo, float2 u2, float2 X2) const;

		//TODO: add threshold constant
		__device__ void regularize() {
			if (m_roughness < 0.3f) {
				m_roughness = clamp(2.0f * m_roughness, 0.1f, 0.3f);
			}
		}

		__device__ bool isNonSpecular() {
			return(m_roughness > Constants::GGX_ROUGHNESS_EPSILON);
		}

		__device__ bool operator! () {
			return m_is_medium;
		}

		__device__ RGBSpectrum getAlbedo() {
			return m_albedo;
		}

	private:
		//BSDFs========================================================
		//Opaque Dielectric BSDF--------------
		__device__ BSDFSample sampleOpaqueDielectric(float3 wo, float2 u2, float X) const;

		__device__ RGBSpectrum fOpaqueDielectric(float3 wo, float3 wi) const;

		__device__ float pdfOpaqueDielectric(float3 wo, float3 wi) const;

		//Conductor BSDF---------
		__device__ BSDFSample sampleConductor(float3 wo, float2 u2, float X) const;

		__device__ RGBSpectrum fConductor(float3 wo, float3 wi) const;

		__device__ float pdfConductor(float3 wo, float3 wi) const;

		//Transparent Dielectric BSDF--------------
		__device__ BSDFSample sampleTransparentDielectric(float3 wo, float2 u2, float X) const;

		__device__ RGBSpectrum fTransparentDielectric(float3 wo, float3 wi) const;

		__device__ float pdfTransparentDielectric(float3 wo, float3 wi) const;

		//BxDFs========================================================
		//diffuse BRDF
		__device__ float3 sampleDiffuseBRDF(float2 u2) const;

		__device__ float fDiffuseBRDF(float3 wo, float3 wi) const;

		__device__ float pdfDiffuseBRDF(float3 wo, float3 wi) const;

		//microfacet glossy BRDF
		__device__ float3 sampleGlossyMicrofacetBRDF_VNDF(float3 wo, float2 u2) const;

		__device__ RGBSpectrum  fGlossyMicrofacetBRDF(float3 wo, float3 wi, float3 h) const;

		__device__ float pdfGlossyMicrofacetBRDF(float3 wo, float3 wi, float3 h) const;

		//microfacet glossy BTDF
		__device__ float pdfGlossyMicrofacetBTDF(float3 wo, float3 wi, float3 ht, float ior) const;

		__device__ RGBSpectrum fGlossyMicrofacetBTDF(float3 wo, float3 wi, float3 ht, float ior) const;

		//---------------------------------------------------------------------------------------------------

	private:
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