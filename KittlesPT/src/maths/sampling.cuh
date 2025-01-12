#pragma once
#include "linear_algebra.cuh"

namespace KittlesPT
{
	__device__ float balanceHeuristic(int nf, float fPdf, int ng, float gPdf);

	__device__ float powerHeuristic(int nf, float fPdf, int ng, float gPdf);

	class RGBSpectrum;
	class IndependentSampler;

	//russian roulette
	__device__ bool russianRoulette(RGBSpectrum* throughput, float eta_scale, int bounce_depth, IndependentSampler& sampler);
}