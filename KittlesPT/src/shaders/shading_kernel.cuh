#pragma once
#include "maths/vector_maths.cuh"
#include <device_launch_parameters.h>

namespace KittlesPT
{
	struct ShadingJob
	{
		int2 pixel_coord;
		float2 uv_coord;
		int2 work_size;
		bool is_invalid = false;
	};

	inline __device__ ShadingJob getShadingJob(int2 work_texture_size) {
		ShadingJob job;

		int thread_pixel_coord_x = threadIdx.x + blockIdx.x * blockDim.x;
		int thread_pixel_coord_y = threadIdx.y + blockIdx.y * blockDim.y;

		job.pixel_coord = make_int2(thread_pixel_coord_x, thread_pixel_coord_y);
		job.work_size = work_texture_size;
		job.uv_coord = (make_float2(job.pixel_coord) + 0.5f) / work_texture_size;//discrete to continous map
		job.is_invalid = ((job.pixel_coord.x >= work_texture_size.x) || (job.pixel_coord.y >= work_texture_size.y));

		return job;
	};
}/*KittlesPT*/