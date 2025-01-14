#include "sampling.cuh"
#include "shaders/color.cuh"
#include "shaders/samplers.cuh"

namespace KittlesPT
{
	__device__ bool russianRoulette(RGBSpectrum* throughput, float eta_scale, int bounce_depth, IndependentSampler& sampler)
	{
		RGBSpectrum rr_beta = *throughput * eta_scale;
		if (rr_beta.maxComponentValue() < 1 && bounce_depth > 1) {
			float q = fmaxf(0.0f, 1.0f - rr_beta.maxComponentValue());
			if (sampler.get1D() < q)
			{
				return true;
			}
			*throughput /= (1.0f - q);
		}
		return false;
	}
}