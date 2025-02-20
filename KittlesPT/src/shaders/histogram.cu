#include "histogram.cuh"

//#define __CUDACC__

#include "color.cuh"
#include "maths/constants.cuh"
#include "maths/linear_algebra.cuh"
#include "containers.cuh"
#include "shading_kernel.cuh"

//#include <device_functions.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>

namespace KittlesPT
{
	// For a given luminance and luminance range, return the histogram bin index
	__device__ uint luminanceToBin(float luminance, float min_log_lum, float inverse_log_lum_range)
	{
		// Avoid taking the log of zero (effectively black pixel)
		if (luminance < Constants::HISTOGRAM_LUMINANCE_EPSILON) {
			return 0u;
		}
		//the surviving darkest pixels are "working black"

		//TODO: clamping normalization here
		// Calculate the log_2 luminance(so EV) and express it as a value in [0.0, 1.0]
		// where 0.0 represents the minimum luminance(EVmin), and 1.0 represents the pmax(EVmax).
		float log_lum_normalized = clamp((log2(luminance) - min_log_lum) * inverse_log_lum_range, 0.0f, 1.0f);//normalization

		// Map [0, 1] to [1, 255]. The zeroth bin is handled by the epsilon check above.
		return uint((log_lum_normalized * (Constants::HISTOGRAM_SIZE - 2)) + 1);//Quantization
	};

	__device__ float centerMeteringWeight(int2 frame_resolution, int2 pixel_coord, float radius_factor)
	{
		int2 centre_spot = frame_resolution / 2.0f;

		float distance = length(make_float2(centre_spot - pixel_coord));
		distance *= (1.0f / radius_factor);

		float max_distance = fminf(frame_resolution.x, frame_resolution.y) / 2.0f;//not using width?
		float normalized_distance = distance / max_distance;

		float weight = 1.0f - smoothstep(0.0f, 1.0f, normalized_distance);

		return clamp(weight, 0.0f, 1.0f);
	};
	//for reference:
	//float LOG_LUM_RANGE = 15.f;
	//float MIN_LOG_LUM = -10.f;
}/*KittlesPT*/

//Launch with thread dims 16x16=256
__global__ void histogramComputeKernel(const KittlesPT::ShaderData shader_data)
{
	using namespace KittlesPT;

	int2 frame_res = shader_data.render_resolution;
	ShadingJob shading_job = getShadingJob(frame_res);
	if (shading_job.is_invalid) {
		return;
	}

	__shared__ float shared_log_lum_histogram[Constants::HISTOGRAM_SIZE];
	int2 local_id = make_int2(threadIdx.x, threadIdx.y);
	int local_index = local_id.x + (local_id.y * blockDim.x);

	{
		//initialize
		shared_log_lum_histogram[local_index] = 0.0f;
		__syncthreads();

		RGBSpectrum linear_radiance = RGBSpectrum(shader_data.output_texture.textureReadNearest(make_float2(shading_job.pixel_coord)));
		float dynamic_range_ev = shader_data.scene_camera.getFilm().white_point_ev - shader_data.scene_camera.getFilm().black_point_ev;
		uint log_lum_bin_idx = luminanceToBin(linear_radiance.getLuminance(),
			shader_data.scene_camera.getFilm().black_point_ev, (1.0f / dynamic_range_ev));

		float weight = centerMeteringWeight(frame_res, shading_job.pixel_coord, 0.4f);//TODO: user param radius
		//weight = 1.0f;
		//shader_data.gbuffer_texture.textureWrite(make_float4(make_float3(weight), 1), shading_job.pixel_coord);

		weight = (log_lum_bin_idx > 0) ? weight : 1;//store pixels count for black pixels(log_lum_bin_idx==0) instead of weights sum

		atomicAdd(&(shared_log_lum_histogram[log_lum_bin_idx]), weight);
		__syncthreads();
	}

	atomicAdd(&(shader_data.histogram_buffer.data[local_index]), shared_log_lum_histogram[local_index]);
}

//launched with thread dims = 256 x 1;
__global__ void histogramAverageLuminanceComputeKernel(const KittlesPT::ShaderData shader_data)
{
	using namespace KittlesPT;

	int2 frame_res = shader_data.render_resolution;
	ShadingJob shading_job = getShadingJob(frame_res);
	if (shading_job.is_invalid) {
		return;
	}

	__shared__ float net_weighted_log_lum_bin_idx[Constants::HISTOGRAM_SIZE];//net meter-weighted log_lum_idx
	__shared__ float net_weights[Constants::HISTOGRAM_SIZE];

	int2 local_id = make_int2(threadIdx.x, threadIdx.y);//basically unused
	int local_index = local_id.x + (local_id.y * blockDim.x);

	float weights_sum_for_this_bin = shader_data.histogram_buffer.data[local_index];//contains black px count for 0 index
	const int log_lum_bin_idx = local_index;//log_lum = ev

	//initialization
	net_weights[log_lum_bin_idx] = weights_sum_for_this_bin;//NOTE: some weights are 1 for black pixels
	net_weighted_log_lum_bin_idx[log_lum_bin_idx] = weights_sum_for_this_bin * log_lum_bin_idx;//this won't have black pixel contribution, since idx will be 0
	__syncthreads();

	// reset the count in the buffer in anticipation of the next pass
	shader_data.histogram_buffer.data[log_lum_bin_idx] = 0.0f;

	// this loop will perform a weighted sum of bin indexes in histogram
#pragma unroll
	for (uint cutoff = (Constants::HISTOGRAM_SIZE / 2); cutoff > 0; cutoff >>= 1) {
		if (uint(local_index) < cutoff) {
			net_weighted_log_lum_bin_idx[local_index] += net_weighted_log_lum_bin_idx[local_index + cutoff];
			net_weights[local_index] += net_weights[local_index + cutoff];
		}

		__syncthreads();
	}

	if (local_index == 0) {
		// Here we take our weighted sum and divide it by the number of pixels
		// that had luminance greater than zero (since the index == 0, we can use count_for_this_bin to find the number of black pixels)

		//weights_sum_for_this_bin is no of black px, for thread 0
		//net_weights[0] is no of black px + valid weights
		//net_weighted_log_lum_bin_idx[0] doesnt have black pixels contribution since their luminance is 0
		float non_black_weight_sum = max(net_weights[0] - weights_sum_for_this_bin, 0.001);//remove black pixel pseudo weights
		float log_lum_bin_idx_avg = net_weighted_log_lum_bin_idx[0] / non_black_weight_sum;

		//printf("Avg bin idx: %d\n", int(log_lum_bin_idx_avg));

		float dynamic_range_ev = shader_data.scene_camera.getFilm().white_point_ev - shader_data.scene_camera.getFilm().black_point_ev;
		// Map from our histogram space to actual luminance
		float avg_luminance = powf(2, (clamp((log_lum_bin_idx_avg - 1.0f) / 
			float(Constants::HISTOGRAM_SIZE - 2), 0.0f, 1.0f) * dynamic_range_ev) + shader_data.scene_camera.getFilm().black_point_ev);

		float lum_last_frame = *(shader_data.scene_average_luminance);
		const float speed = 0.5f;
		float adapted_lum = lerp(lum_last_frame, avg_luminance, 1.0f - expf(-shader_data.frame_delta_ms * speed));
		*(shader_data.scene_average_luminance) = adapted_lum;
	}
}