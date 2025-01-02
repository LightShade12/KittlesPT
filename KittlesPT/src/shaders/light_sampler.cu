#include "light_sampler.cuh"
#include "light.cuh"

namespace KittlesPT
{
	//sample a light
	__device__ SampledLight LightSampler::sample(const float X) const
	{
		//handle empty buffer
		if (lights_buffer_size < 1)
		{
			return SampledLight();
		}

		const int light_index = ::min(int(X * lights_buffer_size), int(lights_buffer_size - 1));

		const Light* sampled_light = &(lights_buffer[light_index]);

		return SampledLight(sampled_light, PMF(sampled_light));
	}

	//probability of sampling the light
	__device__ float LightSampler::PMF(const Light* light) const
	{
		//handle empty buffer
		if (lights_buffer_size < 1)
		{
			return 0;
		}
		return (1.0f / lights_buffer_size);
	}
}