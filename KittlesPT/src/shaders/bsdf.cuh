#pragma once
#include "maths/matrix_maths.cuh"
#include "maths/constants.cuh"

namespace KittlesPT
{
	struct BSDFSample
	{
		__device__ BSDFSample() = default;
		__device__ BSDFSample(int scatter, float3 f, float3 wi, float pdf) :scatter(scatter), f(f), wi(wi), pdf(pdf) {};

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
		float3 f{};
		float3 wi{};
		float pdf = 0;
	};

	struct DeviceMaterial;

	class BSDF {
	public:

		__device__ BSDF(float3 t, float3 b, float3 n) :tangent_matrix(Mat3(t, b, n)) {};

		__device__ BSDF(const Mat3& tangent_basis, float3 albedo, float roughness);

		__device__ float3 f(float3 r_wo, float3 r_wi) const;

		__device__ float pdf(float3 r_wo, float3 r_wi) const;

		__device__ BSDFSample sampleBSDF(float3 w_wo, float2 u2, float2 X2) const;

	private:

		__device__ float3 sampleGlossyMicrofacetBRDF_VNDF(float3 wo, float2 u2) const;

		__device__ float3 fGlossyMicrofacetBRDF(float3 wo, float3 wi, float3 h) const;

		__device__ float pdfGlossyMicrofacetBRDF(float3 wo, float3 wi, float3 h) const;

		//------

		__device__ BSDFSample sampleOpaqueDielectric(float3 wo, float2 u2, float X) const;

		__device__ float3 fOpaqueDielectric(float3 wo, float3 wi) const;

		__device__ float pdfOpaqueDielectric(float3 wo, float3 wi) const;

		//diffuse brdf
		__device__ float3 sampleDiffuseBRDF(float2 u2) const;

		__device__ float fDiffuseBRDF(float3 wo, float3 wi) const;

		__device__ float pdfDiffuseBRDF(float3 wo, float3 wi) const;

	public:
		float3 albedo_factor = make_float3(1, 0, 0);
		float roughness = 0.5f;
		float ior = 1.45f;
		Mat3 tangent_matrix;
	};
}