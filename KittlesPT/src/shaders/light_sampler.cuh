#pragma once
#include "light.cuh"
#include <cuda_runtime.h>

namespace KittlesPT
{
	//Represents a light that has been selected for sampling
	struct SampledLight
	{
		__device__ SampledLight() = default;
		__device__ SampledLight(const Light* light, float p) :light(light), probability(p) {};

		//-------------------------------------

		__device__ operator bool() const
		{
			return (light != nullptr);
		}
		__device__ bool operator !()
		{
			return (light == nullptr);
		}

		//--------------------------------------

		const Light* light = nullptr;
		float probability = 0;
	};

	//--------------------------------------------------------------------------

	//The Uniform light sampler
	class LightSampler
	{
	public:

		__device__ LightSampler(const Light* lights_buffer,
			size_t lights_buffer_size) :
			lights_buffer(lights_buffer),
			lights_buffer_size(lights_buffer_size) {};

		//------------------------------------

		//sample a light
		__device__ SampledLight sample(const float X) const
		{
			//handle empty buffer
			if (lights_buffer_size < 1)
			{
				return SampledLight();
			}

			const int light_index = ::min(int(X * lights_buffer_size), int(lights_buffer_size - 1));

			const Light* sampled_light = &(lights_buffer[light_index]);

			return SampledLight(sampled_light, PMF(sampled_light));
		};

		//probability of sampling the light
		__device__ float PMF(const Light* light) const
		{
			//handle empty buffer
			if (lights_buffer_size < 1)
			{
				return 0;
			}
			return (1.0f / lights_buffer_size);
		}

	private:

		const size_t lights_buffer_size = 0;
		const Light* lights_buffer = nullptr;
	};
}