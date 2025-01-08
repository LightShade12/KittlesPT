#include "sampling.cuh"
#include "shaders/color.cuh"
#include "shaders/samplers.cuh"

namespace KittlesPT
{
	__device__ float balanceHeuristic(int nf, float fPdf, int ng, float gPdf)
	{
		return (nf * fPdf) / (nf * fPdf + ng * gPdf);
	}

	__device__ float powerHeuristic(int nf, float fPdf, int ng, float gPdf)
	{
		float f = nf * fPdf, g = ng * gPdf;
		return Sqr(f) / (Sqr(f) + Sqr(g));
	}

	__device__ bool russianRoulette(RGBSpectrum& throughput, float eta_scale, int bounce_depth, IndependentSampler& sampler)
	{
		RGBSpectrum rr_beta = throughput * eta_scale;
		if (rr_beta.maxComponentValue() < 1 && bounce_depth > 1) {
			float q = fmaxf(0.0f, 1.0f - rr_beta.maxComponentValue());
			if (sampler.get1D() < q)
			{
				return true;
			}
			throughput /= (1.0f - q);
		}
		return false;
	}
}