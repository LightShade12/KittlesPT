#pragma once
#include "maths/matrix_maths.cuh"
#include "maths/constants.cuh"
#include "color.cuh"

namespace KittlesPT
{
	struct BSDFSample
	{
		__device__ BSDFSample() = default;
		__device__ BSDFSample(int scatter, RGBSpectrum f, float3 wi, float pdf) :scatter(scatter), f(f), wi(wi), pdf(pdf) {};

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
		RGBSpectrum f = RGBSpectrum(0);
		float3 wi{};
		float pdf = 0;
	};

	class BSDF
	{
	public:

		__device__ BSDF(float3 t, float3 b, float3 n) :tangent_matrix(Mat3(t, b, n)) {};

		__device__ BSDF(const Mat3& tangent_basis,
			float3 albedo,
			float metallicity,
			float roughness,
			float transmission,
			float ior,
			bool is_backface);

		__device__ RGBSpectrum f(float3 r_wo, float3 r_wi) const;

		__device__ float pdf(float3 r_wo, float3 r_wi) const;

		__device__ BSDFSample sampleBSDF(float3 w_wo, float2 u2, float2 X2) const;

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

		//BRDFs========================================================
		//diffuse brdf
		__device__ float3 sampleDiffuseBRDF(float2 u2) const;

		__device__ float fDiffuseBRDF(float3 wo, float3 wi) const;

		__device__ float pdfDiffuseBRDF(float3 wo, float3 wi) const;

		//microfacet glossy brdf
		__device__ float3 sampleGlossyMicrofacetBRDF_VNDF(float3 wo, float2 u2) const;

		__device__ RGBSpectrum  fGlossyMicrofacetBRDF(float3 wo, float3 wi, float3 h) const;

		__device__ float pdfGlossyMicrofacetBRDF(float3 wo, float3 wi, float3 h) const;

		//microfacet glossy btdf
		__device__ float pdfGlossyMicrofacetBTDF(float3 wo, float3 wi, float3 ht, float ior) const;

		__device__ RGBSpectrum fGlossyMicrofacetBTDF(float3 wo, float3 wi, float3 ht, float ior) const;

	public:
		RGBSpectrum albedo_factor = RGBSpectrum(1, 0, 0);
		float roughness = 0.5f;
		float metallicity = 0.0f;
		float transmission = 0.0f;
		float IOR = 1.45f;
		bool backface = false;

		float alpha = 1.0f;//roughness sq
		Mat3 tangent_matrix;
	};
}