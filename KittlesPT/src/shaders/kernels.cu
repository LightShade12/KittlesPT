#include "kernels.cuh"

#include "error_check.cuh"
#include "maths/linear_algebra.cuh"
#include "ray.cuh"
#include "bsdf.cuh"
#include "samplers.cuh"
#include "sphere.cuh"
#include "integrator.cuh"

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void computePathTraceSamples(const KittlesPT::GlobalShaderData shader_data);

namespace KittlesPT
{
	void launchRenderPassKernel(const GlobalShaderData& shader_data)
	{
		int thread_block_x = 8, thread_block_y = 8;//8x8=64=32x2
		dim3 thread_block_dimensions = dim3(thread_block_x, thread_block_y);
		dim3 thread_block_grid_dimensions = dim3(shader_data.frame_resolution.x / thread_block_x + 1,
			shader_data.frame_resolution.y / thread_block_y + 1);

		computePathTraceSamples << < thread_block_grid_dimensions, thread_block_dimensions >> > (shader_data);
		cudaDeviceSynchronize();
		checkCudaErrors(cudaGetLastError());
	}
}

__global__ void computePathTraceSamples(const KittlesPT::GlobalShaderData shader_data)
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
	sampler.initPixelSeed(pixel_coord, frame_res.x, shader_data.frame_index + 1);//TODO: make sample index internally non-zero

	Ray primary_ray = shader_data.scene_camera.generateRay(ndc_coord, frame_res);

	//evaluate integral(f(x)/p(x)) at Xi
	RGBSpectrum sensor_radiance = Integrator::sensorRadiance(shader_data, primary_ray, sampler);

	//Monte-Carlo estimation; static accumulation
	shader_data.accumulation_texture.textureWrite(
		make_float4(sensor_radiance.toFloat3() + make_float3(shader_data.accumulation_texture.textureReadNearest(pixel_coord)), 1),
		pixel_coord);
	sensor_radiance = RGBSpectrum(shader_data.accumulation_texture.textureReadNearest(pixel_coord)) / ((float)shader_data.frame_index + 1);

	//post process
	sensor_radiance *= shader_data.scene_camera.film.exposure;
	float3 frag_color = shader_data.scene_camera.film.getDisplayRGB(sensor_radiance);
	//frag_color = sensor_radiance / 2.5f;

	shader_data.main_texture.textureWrite(make_float4(frag_color, 1), pixel_coord);
}