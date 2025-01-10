#include "kernels.cuh"

#include "maths/linear_algebra.cuh"
#include "error_check.cuh"

#include "shading_kernel.cuh"
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
}

__global__ void computePathTraceSamples(const KittlesPT::GlobalShaderData shader_data)
{
	using namespace KittlesPT;

	ShadingJob shading_job = getShadingJob(shader_data);

	int2 frame_res = shader_data.frame_resolution;

	if (shading_job.invalid) {
		return;
	}

	float2 ndc_coord = 2.0f * shading_job.uv_coord - 1.0f;
	GBuffer visible_surface;

	IndependentSampler sampler;
	sampler.initPixelSeed(shading_job.pixel_coord, frame_res.x, shader_data.frame_index);

	Filter filter({ 1.0f,1.0f });
	FilterSample fs = filter.sample(sampler.get2D());
	float2 jittered_ndc = ndc_coord + fs.p / (frame_res * 2.0f);

	Ray primary_ray = shader_data.scene_camera.generateRay(jittered_ndc, frame_res);

	float camera_weight = 1.0f;
	//evaluate integral(f(x)/p(x)) at Xi
	RGBSpectrum sensor_radiance = fs.weight * camera_weight * Integrator::Li(shader_data, primary_ray, sampler, &visible_surface);

	float4 packed = visible_surface.packGBuffer();
	shader_data.gbuffer_texture.textureWrite(packed, shading_job.uv_coord);

	//Monte-Carlo estimation; static accumulation
	sensor_radiance = Integrator::addSample(shader_data, shading_job.pixel_coord, sensor_radiance);

	shader_data.main_texture.textureWrite(make_float4(sensor_radiance.toFloat3(), 1), shading_job.uv_coord);
}

__global__ void computePostProcess(const KittlesPT::GlobalShaderData shader_data)
{
	using namespace KittlesPT;

	ShadingJob shading_job = getShadingJob(shader_data);

	if (shading_job.invalid)
	{
		return;
	}

	RGBSpectrum raw_radiance = RGBSpectrum(shader_data.main_texture.textureReadNearest(shading_job.uv_coord));

	if (shader_data.pathtracer_settings.generate_bloom) {
		RGBSpectrum bloom_radiance = RGBSpectrum(shader_data.bloom_texture.textureReadNearest(shading_job.uv_coord));
		raw_radiance = lerp(raw_radiance, bloom_radiance, 0.3f);
	}

	//post process
	raw_radiance *= shader_data.scene_camera.film.exposure;//TODO: proper exposure application
	float3 frag_color = shader_data.scene_camera.film.getDisplayRGB(raw_radiance);

	shader_data.main_texture.textureWrite(make_float4(frag_color, 1), shading_job.uv_coord);
}