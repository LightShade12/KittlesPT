#pragma once
#include <cuda_runtime.h>

namespace KittlesPT
{
	class Light;

	//Represents a light that has been selected for sampling
	struct SampledLight
	{
		SampledLight() = default;
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
	class UniformLightSampler
	{
	public:

		__device__ UniformLightSampler(const Light* lights_buffer,
			size_t lights_buffer_size) :
			lights_buffer(lights_buffer),
			lights_buffer_size(lights_buffer_size) {};

		//------------------------------------

		//sample a light
		__device__ SampledLight sample(const float X) const;

		//probability of sampling the light
		__device__ float PMF(const Light* light) const;

	private:

		const size_t lights_buffer_size = 0;
		const Light* lights_buffer = nullptr;
	};
}/*KittlesPT*/