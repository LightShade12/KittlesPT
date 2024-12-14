#include "kernels.cuh"

#include "../error_check.cuh"
#include "../maths/linear_algebra.cuh"
#include "ray.cuh"
#include "samplers.cuh"
#include "sphere.cuh"

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void renderUV(const KittlesPT::GlobalShaderData shader_data);

namespace KittlesPT
{
	void launchRenderPassKernel(const GlobalShaderData& shader_data)
	{
		int thread_block_x = 8, thread_block_y = 8;//8x8=64=32x2
		dim3 thread_block_dimensions = dim3(thread_block_x, thread_block_y);
		dim3 thread_block_grid_dimensions = dim3(shader_data.frame_resolution.x / thread_block_x + 1,
			shader_data.frame_resolution.y / thread_block_y + 1);

		renderUV << < thread_block_grid_dimensions, thread_block_dimensions >> > (shader_data);

		checkCudaErrors(cudaGetLastError());
	}

	__device__ SurfaceInteraction closestHit(const Ray& ray, const Intersection& intr, const Sphere& sp)
	{
		SurfaceInteraction surfintr;

		surfintr.world_position = ray.getPointAt(intr.distance);
		surfintr.distance = intr.distance;
		surfintr.world_normal = normalize(surfintr.world_position - sp.world_position);

		return surfintr;
	}
}

__global__ void renderUV(const KittlesPT::GlobalShaderData shader_data)
{
	using namespace KittlesPT;
	//setup threads
	int thread_pixel_coord_x = threadIdx.x + blockIdx.x * blockDim.x;
	int thread_pixel_coord_y = threadIdx.y + blockIdx.y * blockDim.y;
	int2 pixel_coord = make_int2(thread_pixel_coord_x, thread_pixel_coord_y);

	int2 frame_res = shader_data.frame_resolution;

	float2 uv_coord = make_float2((float)pixel_coord.x / (float)frame_res.x, (float)pixel_coord.y / (float)frame_res.y);

	if ((pixel_coord.x >= frame_res.x) || (pixel_coord.y >= frame_res.y)) return;
	//============================================
	float2 ndc_coord = uv_coord * 2 - 1;
	IndependentSampler sampler;
	sampler.initPixelSeed(pixel_coord, frame_res.x, shader_data.frame_index);

	Ray primary_ray = shader_data.scene_camera.generateRay(ndc_coord, frame_res);

	float3 frag_color = make_float3(ndc_coord.x, ndc_coord.y, 0.25);

	Intersection closest;
	closest.distance = INFINITY;
	int closest_id = -1;

	for (int id = 0; id < shader_data.scene_buffer.num; id++)
	{
		const Sphere& sphere = shader_data.scene_buffer.data[id];
		Intersection intr = sphere.intersect(primary_ray);
		if (intr.distance < closest.distance && intr.distance >= 0)
		{
			closest = intr;
			closest_id = id;
		}
	}

	if (!closest || closest_id < 0)
	{
		//miss
		float3 unit_direction = normalize(primary_ray.getDirection());
		float a = 0.5 * (unit_direction.y + 1.0);
		frag_color = (1.0 - a) * make_float3(1.0, 1.0, 1.0) + a * make_float3(0.5, 0.7, 1.0);
		//frag_color = make_float3(0, 0.35, 0.45);
	}
	else
	{
		//hit
		SurfaceInteraction surfintr = closestHit(primary_ray, closest, shader_data.scene_buffer.data[closest_id]);
		frag_color = surfintr.world_normal;
	}

	shader_data.main_texture.textureWrite(make_float4(frag_color, 1), pixel_coord);
}