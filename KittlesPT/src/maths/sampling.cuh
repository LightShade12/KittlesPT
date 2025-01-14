#pragma once
#include "linear_algebra.cuh"

namespace KittlesPT
{
	inline __device__ float balanceHeuristic(int nf, float fPdf, int ng, float gPdf)
	{
		return (nf * fPdf) / (nf * fPdf + ng * gPdf);
	}

	inline __device__ float powerHeuristic(int nf, float fPdf, int ng, float gPdf)
	{
		float f = nf * fPdf, g = ng * gPdf;
		return Sqr(f) / (Sqr(f) + Sqr(g));
	}

	class RGBSpectrum;
	class IndependentSampler;

	//russian roulette
	__device__ bool russianRoulette(RGBSpectrum* throughput, float eta_scale, int bounce_depth, IndependentSampler& sampler);
}