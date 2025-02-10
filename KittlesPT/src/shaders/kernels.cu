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

__global__ void computePathTraceSamplesMegaKernel(const KittlesPT::ShaderData shader_data);
__global__ void computePostProcess(const KittlesPT::ShaderData shader_data);

namespace KittlesPT
{
	void launchPathTraceComputeMegaKernel(const ShaderData& shader_data)
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

	void launchPostProcessComputeKernel(const ShaderData& shader_data)
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

	void launchHistogramComputeKernel(const ShaderData& shader_data)
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

	void launchHistogramAverageComputeKernel(const ShaderData& shader_data)
	{
		//256 x 1
		dim3 thread_block_dimensions = dim3(Constants::HISTOGRAM_SIZE, 1, 1);
		dim3 block_grid_dimensions = dim3(1, 1, 1);

		histogramAverageLuminanceComputeKernel << < block_grid_dimensions, thread_block_dimensions >> > (shader_data);

		checkCudaErrors(cudaGetLastError());
		checkCudaErrors(cudaDeviceSynchronize());
	}

	void launchBloomDownSampleComputeKernel(const ShaderData& shader_data, const DeviceTextureBuffer& src,
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

	void launchBloomUpSampleComputeKernel(const ShaderData& shader_data, const DeviceTextureBuffer& src, const DeviceTextureBuffer& dst)
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

	__device__ float2 computeVelocity(const ShaderData& shader_data, const GBuffer& gbuffer, const VBuffer& vbuffer)
	{
		float2 velocity = make_float2(0.0f);
		int32_t curr_mesh_id = vbuffer.instance_id;

		if (curr_mesh_id < 0) {
			return velocity;
		}

		Mat4 curr_viewprojection = shader_data.scene_camera.curr_inv_projection_matrix.inverse() * shader_data.scene_camera.curr_inv_view_matrix.inverse();
		Mat4 prev_viewprojection = shader_data.scene_camera.prev_inv_projection_matrix.inverse() * shader_data.scene_camera.prev_inv_view_matrix.inverse();

		const TriangleMesh& mesh = shader_data.meshes_buffer.data[curr_mesh_id];
		Mat4 curr_model = mesh.curr_inv_model_matrix.inverse();
		Mat4 prev_model = mesh.prev_inv_model_matrix.inverse();
		float3 curr_local_pos = make_float3(mesh.curr_inv_model_matrix * make_float4(gbuffer.wpos, 1));

		float4 curr_clip = curr_viewprojection * make_float4(gbuffer.wpos, 1);
		float4 prev_clip = prev_viewprojection * prev_model * make_float4(curr_local_pos, 1);

		float3 curr_ndc = make_float3(curr_clip) / curr_clip.w;
		float3 prev_ndc = make_float3(prev_clip) / prev_clip.w;

		float2 curr_uv = (make_float2(curr_ndc) + 1.0f) / 2.0f;//[-1,1]=>[0,1]
		float2 prev_uv = (make_float2(prev_ndc) + 1.0f) / 2.0f;

		velocity = curr_uv - prev_uv;

		return velocity;
	}

	//depth2 is sampled depth
	__device__ bool testReprojectedDepth(float depth1, float depth2) {
		const float TEMPORAL_DEPTH_REJECT_THRESHOLD = 0.045f;

		if (fabsf(depth1 - depth2) < (depth2 * TEMPORAL_DEPTH_REJECT_THRESHOLD)) {
			return true;
		}
		return false;
	}

	__device__ bool testReprojection(const ShaderData& shader_data, const GBuffer& gb, float2 prev_pixel_coord, float2 t_current_pix)
	{
		const TriangleMesh& mesh = shader_data.meshes_buffer.data[gb.instance_id];

		float3 curr_local_pos = make_float3(mesh.curr_inv_model_matrix * make_float4(gb.wpos, 1));
		Mat4 prev_model = mesh.prev_inv_model_matrix.inverse();

		//DEPTH HEURISTIC-------------
		GBuffer prev_gb = GBuffer::unpackGBuffer(shader_data.prev_gbuffer_texture.textureReadNearest(prev_pixel_coord));
		float prev_depth = prev_gb.depth;

		float3 prev_camera_origin = make_float3(shader_data.scene_camera.prev_inv_view_matrix.inverse()[3]);
		float3 prev_world_pos = make_float3(prev_model * make_float4(curr_local_pos, 1));//clipspace
		float reprojected_prev_depth = length(prev_camera_origin - prev_world_pos);

		return testReprojectedDepth(reprojected_prev_depth, prev_depth);
	}

	__device__ RGBSpectrum reprojectAccumulate(const ShaderData& shader_data, float2 curr_pixel_coord, const GBuffer& gb, float2 velocity, RGBSpectrum sampled_color)
	{
		//setup threads

		RGBSpectrum final_color = sampled_color;

		int curr_mesh_id = gb.instance_id;

		if (curr_mesh_id < 0) {
			return final_color;
		}

		int2 frame_res = shader_data.frame_resolution;
		//reproject
		float2 current_velocity = velocity;
		float2 pixel_offset = current_velocity * make_float2(frame_res);//map UV value to frame_res(pixel coords)

		float2 prev_pixel_coord = curr_pixel_coord - pixel_offset;

		//new fragment; out of screen
		if (prev_pixel_coord.x < 0 || prev_pixel_coord.x >= frame_res.x ||
			prev_pixel_coord.y < 0 || prev_pixel_coord.y >= frame_res.y)
		{
			return final_color;
		}

		//disocclusion/ reproj failure
		if (testReprojection(shader_data, gb, prev_pixel_coord, curr_pixel_coord))
		{
			return final_color;
		}

		float4 prev_color = shader_data.accumulation_texture.textureReadBilinear(prev_pixel_coord, false);

		float color_history_length = prev_color.w;
		const int32_t MAX_ACCUMULATION_FRAMES = 16;

		float alpha = 1.f / fminf(float(color_history_length + 1), MAX_ACCUMULATION_FRAMES);

		final_color = RGBSpectrum(lerp(make_float3(prev_color), make_float3(final_color), alpha));

		//out----
		return final_color;
	}
}/*KittlesPT*/

/*
* The task is to simulate the process and result of :
	1) Physically Based Light Transport in the scene,
	2) Radiance reception at camera sensor,
	3) Image reconstruction and output from camera
*/

//GPU Kernel to compute a single sample for Monte Carlo Integration of the Rendering Equation (task 1 and 2)
__global__ void computePathTraceSamplesMegaKernel(const KittlesPT::ShaderData shader_data)
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
	RGBSpectrum sensor_radiance = RGBSpectrum(fs.weight) * camera_weight *
		Integrator::Li(shader_data, primary_ray, sampler, &visible_surface);

	VBuffer vb = VBuffer(visible_surface);
	vb.velocity = computeVelocity(shader_data, visible_surface, vb);

	shader_data.gbuffer_texture.textureWriteUV(visible_surface.packGBuffer(), shading_job.uv_coord);
	shader_data.vbuffer_texture.textureWriteUV(vb.packVBuffer(), shading_job.uv_coord);

	if (shader_data.renderer_settings.integrator_use_temporal_accumulation)
	{
		//sensor_radiance = reprojectAccumulate(shader_data,
		//	make_float2(shading_job.pixel_coord), visible_surface, vb.velocity, sensor_radiance);
	}
	else
	{
		//Monte-Carlo estimation; static accumulation
		sensor_radiance = Integrator::addSample(shader_data, shading_job.pixel_coord, sensor_radiance);
	}

	float4 frag_color = make_float4(sensor_radiance.toFloat3(), 1.0f);

	//float scale = centerMeteringWeight(frame_res, shading_job.pixel_coord, 1.0f);
	//scale = ceilf(scale);
	//if (!scale) frag_color += make_float4(1.0f) * length(make_float3(frag_color));

	float3 gas_heat_map = (make_float3(0, 1, 0) * visible_surface.blas_hits * 0.02f) +
		(make_float3(0, 0, 1) * visible_surface.tlas_hits * 0.05f);

	GBuffer prev = GBuffer::unpackGBuffer(shader_data.prev_gbuffer_texture.textureReadNearest(make_float2(shading_job.pixel_coord)));

	shader_data.debug_texture.textureWriteUV(make_float4(make_float3(prev.depth), 1.0f), shading_job.uv_coord);
	shader_data.main_texture.textureWriteUV(frag_color, shading_job.uv_coord);
}

//Conversion from radiance to screen pixels (task 3)
__global__ void computePostProcess(const KittlesPT::ShaderData shader_data)
{
	using namespace KittlesPT;

	ShadingJob shading_job = getShadingJob(shader_data.frame_resolution);

	if (shading_job.invalid) {
		return;
	}

	RGBSpectrum sensor_radiance = RGBSpectrum(shader_data.main_texture.textureReadNearestUV(shading_job.uv_coord));

	if (shader_data.renderer_settings.bloom_generate_bloom) {
		RGBSpectrum bloom_radiance = RGBSpectrum(shader_data.bloom_texture.textureReadNearestUV(shading_job.uv_coord));
		sensor_radiance = lerp(sensor_radiance, bloom_radiance, shader_data.renderer_settings.bloom_final_blend);
	}

	float3 Yxy = sensor_radiance.toYxy();
	Yxy.x *= shader_data.scene_camera.getFilm().luminance_exposure_scalar;//scale scene luminance
	sensor_radiance = RGBSpectrum::fromYxy(Yxy);

	float3 frag_color = shader_data.scene_camera.getFilm().getDisplayNonLinearSRGB(sensor_radiance);
	//frag_color = make_float3(shader_data.gbuffer_texture.textureReadNearest(make_float2(shading_job.pixel_coord)));
	//non-linear srgb target; expects gamma correction
	shader_data.main_texture.textureWriteUV(make_float4(frag_color, 1.0f), shading_job.uv_coord);
}