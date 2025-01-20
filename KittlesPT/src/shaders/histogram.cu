#include "histogram.cuh"

#define __CUDACC__

#include "color.cuh"
#include "maths/constants.cuh"
#include "maths/linear_algebra.cuh"
#include "containers.cuh"
#include "shading_kernel.cuh"

#include <device_functions.h>
//#include <cuda_runtime.h>
#include <cooperative_groups.h>

namespace KittlesPT
{
	// For a given color and luminance range, return the histogram bin index
	__device__ uint colorToBin(float3 hdrColor, float minLogLum, float inverseLogLumRange)
	{
		float lum = RGBSpectrum(hdrColor).getLuminance();

		// Avoid taking the log of zero
		if (lum < Constants::HISTOGRAM_LUMINANCE_EPSILON) {
			return 0;
		}

		// Calculate the log_2 luminance and express it as a value in [0.0, 1.0]
		// where 0.0 represents the minimum luminance, and 1.0 represents the max.
		float log_lum = clamp((log2(lum) - minLogLum) * inverseLogLumRange, 0.0, 1.0);

		// Map [0, 1] to [1, 255]. The zeroth bin is handled by the epsilon check above.
		return uint(log_lum * 254.0 + 1.0);
	};

	__device__ float meteringWeight(const GlobalShaderData& shader_data, int2 pixel_coord)
	{
		int2 centre_spot = shader_data.frame_resolution / 2.0f;

		constexpr float radius_factor = 1.0f;//TODO: can be user param

		float distance = length(make_float2(centre_spot - pixel_coord));
		distance *= (1.0f / radius_factor);

		float max_distance = fminf(shader_data.frame_resolution.x, shader_data.frame_resolution.y) / 2.0f;//not using width?
		float normalized_distance = distance / max_distance;

		float weight = 1.0f - smoothstep(0.0f, 1.0f, normalized_distance);

		return clamp(weight, 0.0f, 1.0f);
	};
	//__constant__ constexpr float LOG_LUM_RANGE = 15.f;//TODO: what values for these?
	//__constant__ constexpr float MIN_LOG_LUM = -10.f;
}/*KittlesPT*/

//Launch with thread dims 16x16=256
__global__ void histogramComputeKernel(const KittlesPT::GlobalShaderData shader_data)
{
	using namespace KittlesPT;

	int2 frame_res = shader_data.frame_resolution;
	ShadingJob shading_job = getShadingJob(frame_res);

	if (shading_job.invalid) {
		return;
	}

	__shared__ float shared_histogram[Constants::HISTOGRAM_SIZE];

	int2 local_id = make_int2(threadIdx.x, threadIdx.y);
	int local_index = local_id.x + (local_id.y * blockDim.x);

	{
		shared_histogram[local_index] = 0;//initialize working buffer
		__syncthreads();

		float3 linear_radiance = make_float3(shader_data.main_texture.textureReadNearest(make_float2(shading_job.pixel_coord)));
		float dynamic_range = shader_data.scene_camera.film.white_point - shader_data.scene_camera.film.black_point;
		uint bin_idx = colorToBin(linear_radiance, shader_data.scene_camera.film.black_point, (1.0f / dynamic_range));

		float weight = meteringWeight(shader_data, shading_job.pixel_coord);

		//shader_data.gbuffer_texture.textureWrite(make_float4(make_float3(weight), 1), shading_job.pixel_coord);

		weight = bin_idx > 0 ? weight : 1;//store pixels count for black pixels(bin_idx==0) instead of weights sum

		atomicAdd(&(shared_histogram[bin_idx]), weight);

		__syncthreads();

		atomicAdd(&(shader_data.histogram_buffer.data[local_index]), shared_histogram[local_index]);
	}
}

//launch with thread dims= 256 x 1;
__global__ void histogramAverageLuminanceComputeKernel(const KittlesPT::GlobalShaderData shader_data)
{
	using namespace KittlesPT;

	int2 frame_res = shader_data.frame_resolution;
	ShadingJob shading_job = getShadingJob(frame_res);

	if (shading_job.invalid) {
		return;
	}

	__shared__ float shared_weighted_luminance_histogram[Constants::HISTOGRAM_SIZE];
	__shared__ float shared_net_weights[Constants::HISTOGRAM_SIZE];

	{
		int2 local_id = make_int2(threadIdx.x, threadIdx.y);
		int local_index = local_id.x + (local_id.y * blockDim.x);

		// Get the count from the histogram buffer
		float weights_sum_for_this_bin = shader_data.histogram_buffer.data[local_index];
		const int& bin_luminance_value = local_index;

		shared_weighted_luminance_histogram[local_index] = weights_sum_for_this_bin * bin_luminance_value;//net metering weighted luminance in the bin
		shared_net_weights[local_index] = weights_sum_for_this_bin;

		__syncthreads();

		// Reset the count stored in the buffer in anticipation of the next pass
		shader_data.histogram_buffer.data[local_index] = 0;

		// This loop will perform a weighted count of the luminance range
#pragma unroll
		for (uint cutoff = (Constants::HISTOGRAM_SIZE / 2); cutoff > 0; cutoff >>= 1) {
			if (uint(local_index) < cutoff) {
				shared_weighted_luminance_histogram[local_index] += shared_weighted_luminance_histogram[local_index + cutoff];
				shared_net_weights[local_index] += shared_net_weights[local_index + cutoff];
			}

			__syncthreads();
		}

		// We only need to calculate this once, so only a single thread is needed.
		if (local_index == 0) {
			// Here we take our weighted sum and divide it by the number of pixels
			// that had luminance greater than zero (since the index == 0, we can
			// use count_for_this_bin to find the number of black pixels)
			//float denom = max((frame_res.x * frame_res.y) - float(weights_sum_for_this_bin), 1.0);//works cuz we dont store weight sum for black px

			//weights_sum_for_this_bin is no of black px for thread 0
			//shared_net_weights[0] is no of black px + valid weights
			//shared_weighted_luminance_histogram[0] doesnt have black pixels since their luminance is 0
			float denom = max(shared_net_weights[0] - weights_sum_for_this_bin, 1.0);//works cuz we dont store weight sum for black px
			float weighted_log_average = (shared_weighted_luminance_histogram[0] / denom) - 1.0;

			float dynamic_range = shader_data.scene_camera.film.white_point - shader_data.scene_camera.film.black_point;
			// Map from our histogram space to actual luminance
			float weighted_avg_lum = exp2f(((weighted_log_average / 254.0) * dynamic_range) + shader_data.scene_camera.film.black_point);

			float lum_last_frame = *(shader_data.scene_average_luminance);
			float speed = 1.0f;
			float adapted_lum = lerp(lum_last_frame, weighted_avg_lum, 1.0f - expf(-shader_data.frame_delta * speed));
			*(shader_data.scene_average_luminance) = adapted_lum;
		}
	}
}