#include "shading_kernel.cuh"
#include "maths/vector_maths.cuh"
#include "containers.cuh"

namespace KittlesPT
{
	__device__ ShadingJob getShadingJob(const GlobalShaderData& shader_data)
	{
		ShadingJob job;

		int thread_pixel_coord_x = threadIdx.x + blockIdx.x * blockDim.x;
		int thread_pixel_coord_y = threadIdx.y + blockIdx.y * blockDim.y;

		job.pixel_coord = make_int2(thread_pixel_coord_x, thread_pixel_coord_y);

		int2 frame_res = shader_data.frame_resolution;

		job.uv_coord = (make_float2(job.pixel_coord) + 0.5f) / frame_res;//discrete to continous map

		job.invalid = ((job.pixel_coord.x >= frame_res.x) || (job.pixel_coord.y >= frame_res.y));

		return job;
	}
}