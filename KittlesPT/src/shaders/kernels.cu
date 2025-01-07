#include "kernels.cuh"

#include "maths/linear_algebra.cuh"
#include "error_check.cuh"

#include "containers.cuh"
#include "ray.cuh"
#include "samplers.cuh"
#include "color.cuh"
#include "integrator.cuh"
#include "packing.cuh"
#include "filter.cuh"
#include "bloom.cuh"

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void computePathTraceSamples(const KittlesPT::GlobalShaderData shader_data);
__global__ void computePostProcess(const KittlesPT::GlobalShaderData shader_data);

namespace KittlesPT
{
	void launchPathTraceComputeKernel(const GlobalShaderData& shader_data)
	{
		int thread_block_x = 8, thread_block_y = 8;//8x8=64=32x2
		dim3 thread_block_dimensions = dim3(thread_block_x, thread_block_y);
		dim3 thread_block_grid_dimensions = dim3(shader_data.frame_resolution.x / thread_block_x + 1,
			shader_data.frame_resolution.y / thread_block_y + 1);

		computePathTraceSamples << < thread_block_grid_dimensions, thread_block_dimensions >> > (shader_data);
		cudaDeviceSynchronize();
		checkCudaErrors(cudaGetLastError());
	}

	void launchPostProcessComputeKernel(const GlobalShaderData& shader_data)
	{
		int thread_block_x = 8, thread_block_y = 8;//8x8=64=32x2
		dim3 thread_block_dimensions = dim3(thread_block_x, thread_block_y);
		dim3 thread_block_grid_dimensions = dim3(shader_data.frame_resolution.x / thread_block_x + 1,
			shader_data.frame_resolution.y / thread_block_y + 1);

		computePostProcess << < thread_block_grid_dimensions, thread_block_dimensions >> > (shader_data);
		cudaDeviceSynchronize();
		checkCudaErrors(cudaGetLastError());
	}

	void launchBloomDownSampleComputeKernel(const GlobalShaderData& shader_data, const DeviceTextureBuffer& src, const DeviceTextureBuffer& dst)
	{
		int thread_block_x = 8, thread_block_y = 8;//8x8=64=32x2
		dim3 thread_block_dimensions = dim3(thread_block_x, thread_block_y);
		dim3 thread_block_grid_dimensions = dim3(shader_data.frame_resolution.x / thread_block_x + 1,
			shader_data.frame_resolution.y / thread_block_y + 1);

		downSample << < thread_block_grid_dimensions, thread_block_dimensions >> > (shader_data, src, dst);
		cudaDeviceSynchronize();
		checkCudaErrors(cudaGetLastError());
	}

	void launchBloomUpSampleComputeKernel(const GlobalShaderData& shader_data, const DeviceTextureBuffer& src, const DeviceTextureBuffer& dst, bool combine)
	{
		int thread_block_x = 8, thread_block_y = 8;//8x8=64=32x2
		dim3 thread_block_dimensions = dim3(thread_block_x, thread_block_y);
		dim3 thread_block_grid_dimensions = dim3(shader_data.frame_resolution.x / thread_block_x + 1,
			shader_data.frame_resolution.y / thread_block_y + 1);

		upSampleCombine << < thread_block_grid_dimensions, thread_block_dimensions >> > (shader_data, src, dst, combine);
		cudaDeviceSynchronize();
		checkCudaErrors(cudaGetLastError());
	}

	//Monte-Carlo estimation; static accumulation
	__device__ RGBSpectrum addSample(const GlobalShaderData& shader_data, int2 pixel_coord, RGBSpectrum radiance)
	{
		RGBSpectrum accumulated_sample = RGBSpectrum(shader_data.accumulation_texture.textureReadNearest(pixel_coord));
		RGBSpectrum new_accumulated_sample = accumulated_sample + radiance;

		shader_data.accumulation_texture.textureWrite(make_float4(new_accumulated_sample.toFloat3(), 1), pixel_coord);
		RGBSpectrum sensor_radiance_estimate = new_accumulated_sample / ((float)shader_data.frame_index + 1);

		return sensor_radiance_estimate;
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

	if ((pixel_coord.x >= frame_res.x) || (pixel_coord.y >= frame_res.y)) {
		return;
	}
	//============================================
	float2 ndc_coord = uv_coord * 2 - 1;
	IndependentSampler sampler;
	GBuffer visible_surface;
	Filter filter;

	sampler.initPixelSeed(pixel_coord, frame_res.x, shader_data.frame_index + 1);//TODO: make sample index internally non-zero

	FilterSample fs = filter.sample(sampler.get2D());
	float2 jittered_ndc = ndc_coord + make_float2(0.5f / (float)frame_res.x, 0.5f / (float)frame_res.y);//discrete to continous map
	jittered_ndc += make_float2(fs.p.x / (float)frame_res.x, fs.p.y / (float)frame_res.y);

	Ray primary_ray = shader_data.scene_camera.generateRay(jittered_ndc, frame_res);

	float camera_weight = 1.0f;
	//evaluate integral(f(x)/p(x)) at Xi
	RGBSpectrum sensor_radiance = camera_weight * Integrator::Li(shader_data, primary_ray,
		sampler, &visible_surface);

	//visible surface to GBuffer Film
	float4 packed = packGBuffer(visible_surface);
	shader_data.gbuffer_texture.textureWrite(packed, pixel_coord);

	sensor_radiance.clampOutput();
	//Monte-Carlo estimation; static accumulation
	sensor_radiance = addSample(shader_data, pixel_coord, (sensor_radiance * fs.weight));

	shader_data.main_texture.textureWrite(make_float4(sensor_radiance.toFloat3(), 1), pixel_coord);
}

__global__ void computePostProcess(const KittlesPT::GlobalShaderData shader_data)
{
	using namespace KittlesPT;
	//setup threads
	int thread_pixel_coord_x = threadIdx.x + blockIdx.x * blockDim.x;
	int thread_pixel_coord_y = threadIdx.y + blockIdx.y * blockDim.y;
	int2 pixel_coord = make_int2(thread_pixel_coord_x, thread_pixel_coord_y);

	int2 frame_res = shader_data.frame_resolution;

	float2 uv_coord = make_float2((float)pixel_coord.x / (float)frame_res.x, (float)pixel_coord.y / (float)frame_res.y);

	if ((pixel_coord.x >= frame_res.x) || (pixel_coord.y >= frame_res.y)) {
		return;
	}
	//========================================================================================================================

	RGBSpectrum raw_radiance = RGBSpectrum(shader_data.main_texture.textureReadNearest(pixel_coord));
	RGBSpectrum bloom_radiance = RGBSpectrum(shader_data.bloom_texture.textureReadNearest(pixel_coord));
	//bloom lerp=0.3f;
	raw_radiance = RGBSpectrum(lerp(raw_radiance.toFloat3(), bloom_radiance.toFloat3(), 0.3));

	//post process
	raw_radiance *= shader_data.scene_camera.film.exposure;//TODO: proper exposure application
	float3 frag_color = shader_data.scene_camera.film.getDisplayRGB(raw_radiance);

	shader_data.main_texture.textureWrite(make_float4(frag_color, 1), pixel_coord);
}