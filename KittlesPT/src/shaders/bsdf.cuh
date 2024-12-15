#pragma once
#include "maths/matrix_maths.cuh"
#include "maths/constants.cuh"

namespace KittlesPT
{
	struct BSDFSample 
	{
		__device__ BSDFSample() = default;
		__device__ BSDFSample(float3 f, float3 wi, float pdf) :f(f), wi(wi), pdf(pdf) {};
		float3 f{};
		float3 wi{};
		float pdf = 0;
	};

	struct DeviceMaterial;

	class BSDF {
	public:

		__device__ BSDF(float3 x, float3 y, float3 z) :tangent_matrix(Mat3(x, y, z)) {};

		//BSDF(float3 tang, float3 bitan, float3 n) :tangentMatrix(Mat3(tang, bitan, n)) {};

		__device__ BSDF(const Mat3& tangent_basis, float3 albedo);

		__device__ float3 f(float3 r_wo, float3 r_wi) const;

		__device__ float pdf(float3 r_wo, float3 r_wi) const;

		__device__ BSDFSample sampleBSDF(float3 r_wo, float2 u2) const;

	private:

		__device__ float3 sampleDiffuseBRDF(float2 u2) const;

		__device__ float fDiffuseBRDF(float3 wo,  float3 wi) const;

		__device__ float pdfDiffuseBRDF(float3 wo, float3 wi) const;

		__device__ BSDFSample sampleOpaqueDielectric(float3 wo, float2 u2) const;

		__device__ float3 fOpaqueDielectric(float3 wo, float3 wi) const;

		__device__ float pdfOpaqueDielectric(float3 wo, float3 wi) const;

	public:
		float3 albedo_factor = make_float3(1, 0, 0);

		Mat3 tangent_matrix;
	};
}