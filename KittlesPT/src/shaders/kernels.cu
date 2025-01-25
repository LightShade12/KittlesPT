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
#include "histogram.cuh"

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void computePathTraceSamplesMegaKernel(const KittlesPT::GlobalShaderData shader_data);
__global__ void computePostProcess(const KittlesPT::GlobalShaderData shader_data);

namespace KittlesPT
{
	void launchPathTraceComputeMegaKernel(const GlobalShaderData& shader_data)
	{
		//8 x 8 = 64 = 32 x 2
		dim3 thread_block_dimensions = dim3(8, 8, 1);
		dim3 block_grid_dimensions = dim3(
			shader_data.frame_resolution.x / thread_block_dimensions.x + 1,
			shader_data.frame_resolution.y / thread_block_dimensions.y + 1,
			1);

		computePathTraceSamplesMegaKernel << < block_grid_dimensions, thread_block_dimensions >> > (shader_data);
		checkCudaErrors(cudaGetLastError());
		checkCudaErrors(cudaDeviceSynchronize());
	}

	void launchPostProcessComputeKernel(const GlobalShaderData& shader_data)
	{
		//8 x 8 = 64 = 32 x 2
		dim3 thread_block_dimensions = dim3(8, 8, 1);
		dim3 block_grid_dimensions = dim3(
			shader_data.frame_resolution.x / thread_block_dimensions.x + 1,
			shader_data.frame_resolution.y / thread_block_dimensions.y + 1,
			1);

		computePostProcess << < block_grid_dimensions, thread_block_dimensions >> > (shader_data);
		checkCudaErrors(cudaGetLastError());
		checkCudaErrors(cudaDeviceSynchronize());
	}

	void launchHistogramComputeKernel(const GlobalShaderData& shader_data)
	{
		//16x16 = 256
		dim3 thread_block_dimensions = dim3(16, 16, 1);
		dim3 block_grid_dimensions = dim3(
			shader_data.frame_resolution.x / thread_block_dimensions.x + 1,
			shader_data.frame_resolution.y / thread_block_dimensions.y + 1,
			1);

		histogramComputeKernel << < block_grid_dimensions, thread_block_dimensions >> > (shader_data);

		checkCudaErrors(cudaGetLastError());
		checkCudaErrors(cudaDeviceSynchronize());
	}

	void launchHistogramAverageComputeKernel(const GlobalShaderData& shader_data)
	{
		//256 x 1
		dim3 thread_block_dimensions = dim3(Constants::HISTOGRAM_SIZE, 1, 1);
		dim3 block_grid_dimensions = dim3(1, 1, 1);

		histogramAverageLuminanceComputeKernel << < block_grid_dimensions, thread_block_dimensions >> > (shader_data);

		checkCudaErrors(cudaGetLastError());
		checkCudaErrors(cudaDeviceSynchronize());
	}

	void launchBloomDownSampleComputeKernel(const GlobalShaderData& shader_data, const DeviceTextureBuffer& src,
		const DeviceTextureBuffer& dst, bool karis_avg)
	{
		//8x8=64=32x2
		dim3 thread_block_dimensions = dim3(8, 8, 1);
		dim3 block_grid_dimensions = dim3(
			dst.dimensions.x / thread_block_dimensions.x + 1,
			dst.dimensions.y / thread_block_dimensions.y + 1,
			1);

		downSample << < block_grid_dimensions, thread_block_dimensions >> > (shader_data, src, dst, karis_avg);

		checkCudaErrors(cudaGetLastError());
		checkCudaErrors(cudaDeviceSynchronize());
	}

	void launchBloomUpSampleComputeKernel(const GlobalShaderData& shader_data, const DeviceTextureBuffer& src, const DeviceTextureBuffer& dst)
	{
		//8x8=64=32x2
		dim3 thread_block_dimensions = dim3(8, 8, 1);
		dim3 block_grid_dimensions = dim3(
			dst.dimensions.x / thread_block_dimensions.x + 1,
			dst.dimensions.y / thread_block_dimensions.y + 1,
			1);

		upSampleCombine << < block_grid_dimensions, thread_block_dimensions >> > (shader_data, src, dst);

		checkCudaErrors(cudaGetLastError());
		checkCudaErrors(cudaDeviceSynchronize());
	}
}/*KittlesPT*/

__global__ void computePathTraceSamplesMegaKernel(const KittlesPT::GlobalShaderData shader_data)
{
	using namespace KittlesPT;

	int2 frame_res = shader_data.frame_resolution;
	ShadingJob shading_job = getShadingJob(frame_res);

	if (shading_job.invalid) {
		return;
	}

	IndependentSampler sampler;
	sampler.initPixelSeed(shading_job.pixel_coord, frame_res.x, shader_data.frame_index);

	BoxFilter filter({ 1.0f,1.0f });
	FilterSample fs = filter.sample(sampler.get2D());

	float2 ndc_coord = 2.0f * shading_job.uv_coord - 1.0f;
	float2 jittered_ndc = ndc_coord + fs.p / (frame_res * 2.0f);

	Ray primary_ray = shader_data.scene_camera.generateRay(jittered_ndc);

	GBuffer visible_surface;
	float camera_weight = 1.0f;
	//We estimate radiance directly as RGB triplets
	//evaluate integral(f(x)/p(x)) at Xi
	RGBSpectrum sensor_radiance = fs.weight * camera_weight * Integrator::Li(shader_data,
		primary_ray, sampler, &visible_surface);

	shader_data.gbuffer_texture.textureWriteUV(visible_surface.packGBuffer(), shading_job.uv_coord);

	//Monte-Carlo estimation; static accumulation
	sensor_radiance = Integrator::addSample(shader_data, shading_job.pixel_coord, sensor_radiance);

	float4 frag_color = make_float4(sensor_radiance.toFloat3(), 1.0f);

	//float scale = centerMeteringWeight(frame_res, shading_job.pixel_coord, 1.0f);
	//scale = ceilf(scale);
	//if (!scale) frag_color += make_float4(1.0f) * length(make_float3(frag_color));

	shader_data.main_texture.textureWriteUV(frag_color, shading_job.uv_coord);
}

__global__ void computePostProcess(const KittlesPT::GlobalShaderData shader_data)
{
	using namespace KittlesPT;

	ShadingJob shading_job = getShadingJob(shader_data.frame_resolution);

	if (shading_job.invalid) {
		return;
	}

	RGBSpectrum sensor_radiance = RGBSpectrum(shader_data.main_texture.textureReadNearestUV(shading_job.uv_coord));

	if (shader_data.renderer_settings.generate_bloom) {
		RGBSpectrum bloom_radiance = RGBSpectrum(shader_data.bloom_texture.textureReadNearestUV(shading_job.uv_coord));
		sensor_radiance = lerp(sensor_radiance, bloom_radiance, shader_data.renderer_settings.bloom_blend);
	}

	float3 Yxy = sensor_radiance.toYxy();
	Yxy.x *= shader_data.scene_camera.film.luminance_exposure_scalar;//scale scene luminance
	sensor_radiance = RGBSpectrum::fromYxy(Yxy);

	float3 frag_color = shader_data.scene_camera.film.getDisplayNonLinearSRGB(sensor_radiance);
	//frag_color = make_float3(shader_data.gbuffer_texture.textureReadNearest(make_float2(shading_job.pixel_coord)));
	//non-linear srgb target; expects gamma correction
	shader_data.main_texture.textureWriteUV(make_float4(frag_color, 1.0f), shading_job.uv_coord);
}