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
__global__ void computeEffects(const KittlesPT::ShaderData shader_data);
__global__ void modulateSamples(const KittlesPT::ShaderData shader_data);

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

	void launchFxComputeKernel(const ShaderData& shader_data)
	{
		//8 x 8 = 64 = 32 x 2
		dim3 thread_block_dimensions = dim3(8, 8, 1);
		dim3 block_grid_dimensions = dim3(
			shader_data.frame_resolution.x / thread_block_dimensions.x + 1,
			shader_data.frame_resolution.y / thread_block_dimensions.y + 1,
			1);

		computeEffects << < block_grid_dimensions, thread_block_dimensions >> > (shader_data);
		checkCudaErrors(cudaGetLastError());
		checkCudaErrors(cudaDeviceSynchronize());
	}

	void launchModulateComputeKernel(const ShaderData& shader_data)
	{
		//8 x 8 = 64 = 32 x 2
		dim3 thread_block_dimensions = dim3(8, 8, 1);
		dim3 block_grid_dimensions = dim3(
			shader_data.frame_resolution.x / thread_block_dimensions.x + 1,
			shader_data.frame_resolution.y / thread_block_dimensions.y + 1,
			1);

		modulateSamples << < block_grid_dimensions, thread_block_dimensions >> > (shader_data);
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

	__device__ float2 computeVelocity(const ShaderData& shader_data, const VisibleSurface& vs)
	{
		float2 velocity = make_float2(0.0f);
		int32_t curr_mesh_id = vs.instance_id;

		if (curr_mesh_id < 0) {
			return velocity;
		}

		Mat4 curr_viewprojection = shader_data.scene_camera.curr_inv_projection_matrix.inverse() * shader_data.scene_camera.curr_inv_view_matrix.inverse();
		Mat4 prev_viewprojection = shader_data.scene_camera.prev_inv_projection_matrix.inverse() * shader_data.scene_camera.prev_inv_view_matrix.inverse();

		const TriangleMesh& mesh = shader_data.meshes_buffer.data[curr_mesh_id];
		Mat4 curr_model = mesh.curr_inv_model_matrix.inverse();
		Mat4 prev_model = mesh.prev_inv_model_matrix.inverse();
		float3 curr_local_pos = make_float3(mesh.curr_inv_model_matrix * make_float4(vs.position, 1));

		float4 curr_clip = curr_viewprojection * make_float4(vs.position, 1);
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
		const float TEMPORAL_DEPTH_REJECT_THRESHOLD = 0.045f; //TODO: put in a constants file

		if (fabsf(depth1 - depth2) < (depth2 * TEMPORAL_DEPTH_REJECT_THRESHOLD)) {
			return true;
		}
		return false;
	}

	__device__ bool testReprojection(const ShaderData& shader_data, const VisibleSurface& vs, float2 prev_pixel_coord)
	{
		const TriangleMesh& mesh = shader_data.meshes_buffer.data[vs.instance_id];

		float3 curr_local_pos = make_float3(mesh.curr_inv_model_matrix * make_float4(vs.position, 1));
		Mat4 prev_model = mesh.prev_inv_model_matrix.inverse();

		//DEPTH HEURISTIC-------------
		GBuffer prev_gb = GBuffer::unpackGBuffer(shader_data.prev_gbuffer_texture.textureReadNearest(prev_pixel_coord));
		float prev_depth = prev_gb.depth;

		float3 prev_camera_origin = make_float3(shader_data.scene_camera.prev_inv_view_matrix[3]);
		float3 prev_world_pos = make_float3(prev_model * make_float4(curr_local_pos, 1));//clipspace
		float reprojected_prev_depth = length(prev_camera_origin - prev_world_pos);

		return testReprojectedDepth(reprojected_prev_depth, prev_depth);
	}

	__device__ bool resamplePrevColor(float2 pixel_coord, float4& fc, const ShaderData& shader_data, const VisibleSurface& vs)
	{
		bool vt[4];
		int2 offsets[4] = { {0,0},{1,0},{0,1},{1,1} };
		bool valid = false;
		int2 ipos = make_int2(pixel_coord);

		for (int32_t sample_idx = 0; sample_idx < 4; sample_idx++) {
			float2 tap = pixel_coord + offsets[sample_idx];
			vt[sample_idx] = testReprojection(shader_data, vs, tap);
			valid |= vt[sample_idx];
		}
		if (valid) {
			float sumw = 0.0f;
			float x = fracf(pixel_coord.x);
			float y = fracf(pixel_coord.y);

			// bilinear weights
			float w[4] = { (1 - x) * (1 - y),
								x * (1 - y),
						   (1 - x) * y,
								x * y };

			fc = make_float4(0, 0, 0, 0);

			// perform the actual bilinear interpolation
			for (int32_t sample_idx = 0; sample_idx < 4; sample_idx++)
			{
				float2 tap = pixel_coord + offsets[sample_idx];
				if (vt[sample_idx]) {
					fc += w[sample_idx] * shader_data.accumulation_texture.textureReadNearest(tap);
					sumw += w[sample_idx];
				}
			}

			// redistribute weights in case not all taps were used
			valid = (sumw >= 0.01);
			fc = (valid) ? fc / sumw : make_float4(0, 0, 0, 0);
		}

		//3x3 backup resample

		if (!valid) // perform cross-bilateral filter in the hope to find some suitable samples somewhere
		{
			float cnt = 0.0;

			// this code performs a binary descision for each tap of the cross-bilateral filter
			const int32_t radius = 1;
			for (int32_t yy = -radius; yy <= radius; yy++) {
				for (int32_t xx = -radius; xx <= radius; xx++) {
					int2 tap = ipos + make_int2(xx, yy);
					if (testReprojection(shader_data, vs, make_float2(tap))) {
						fc += shader_data.accumulation_texture.textureReadNearest(make_float2(tap));
						cnt += 1.0f;
					}
				}
			}
			if (cnt > 0) {
				valid = true;
				fc /= cnt;
			}
		}

		if (valid) {
			fc.w = shader_data.accumulation_texture.textureReadNearest(pixel_coord).w;
		}
		else {
			fc = make_float4(0);
		}

		return valid;
	}

	__constant__ constexpr int32_t MAX_ACCUMULATION_FRAMES = 16;//TODO: put in constants
	//#define SUPER_SAMPLE
	__device__ float4 reprojectAccumulate(const ShaderData& shader_data, float2 curr_pixel_coord, const VisibleSurface& vs,
		float2 velocity, RGBSpectrum curr_color)
	{
		int32_t curr_mesh_id = vs.instance_id;
		//no surface
		if (curr_mesh_id < 0) {
			return make_float4(curr_color.toFloat3(), 0);
		}

		int2 frame_res = shader_data.frame_resolution;
		//reproject
		float2 delta_pixel_coord = velocity * make_float2(frame_res);//map UV value to frame_res(pixel coords)
		float2 prev_pixel_coord = curr_pixel_coord - delta_pixel_coord;

		//new out of screen fragment;
		if (prev_pixel_coord.x < 0 || prev_pixel_coord.x >= frame_res.x ||
			prev_pixel_coord.y < 0 || prev_pixel_coord.y >= frame_res.y) {
			return make_float4(curr_color.toFloat3(), 0);
		}

#ifndef SUPER_SAMPLE
		//disocclusion/ reprojection failure
		if (!testReprojection(shader_data, vs, prev_pixel_coord)) {
			return make_float4(curr_color.toFloat3(), 0);
		}

		float4 prev_color = shader_data.accumulation_texture.textureReadBilinear(prev_pixel_coord, false);
#else
		float4 prev_color;
		if (!resamplePrevColor(prev_pixel_coord, prev_color, shader_data, vs)) {
			return make_float4(curr_color.toFloat3(), 0);
		}
#endif // !SUPER_SAMPLE

		float color_history_length = prev_color.w;
		float alpha = 1.0f / fminf(color_history_length + 1, (float)MAX_ACCUMULATION_FRAMES);
		RGBSpectrum final_color = lerp(RGBSpectrum(prev_color), curr_color, alpha);

		//out----
		return make_float4(final_color.toFloat3(), color_history_length + 1);
	}

	__device__ float2 getFishEyeUV(float2 pixel_coord, float2 frame_res)
	{
		float2 p = pixel_coord / frame_res.x;//normalized coords with some cheat
		//(assume 1:1 prop)
		float prop = frame_res.x / frame_res.y;//screen proroption
		float2 m = make_float2(0.5, 0.5 / prop);//center coords
		float2 d = p - m;//vector from center to current fragment
		float r = sqrt(dot(d, d)); // distance of pixel from center

		float power = 1.65f;

		float bind;//radius of 1:1 effect
		if (power > 0.0) {
			bind = sqrt(dot(m, m));//stick to corners
		}
		else {
			if (prop < 1.0) {
				bind = m.x;
			}
			else {
				bind = m.y;
			}
		}//stick to borders

//Weird formulas
		float2 uv;
		if (power > 0.0)//fisheye
			uv = m + normalize(d) * tan(r * power) * bind / tan(bind * power);
		else if (power < 0.0)//antifisheye
			uv = m + normalize(d) * atan(r * -power * 10.0) * bind / atan(-power * bind * 10.0);
		else uv = p;//no effect for power = 1.0

		uv.y *= prop;
		return uv;
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
	jittered_ndc = ndc_coord;

	DebugData debugdata;
	Ray primary_ray = shader_data.scene_camera.generateRay(jittered_ndc);
	VisibleSurface visible_surface;
	float camera_weight = 1.0f;
	//We estimate radiance directly as RGB triplets
	//evaluate integral(f(x)/p(x)) at Xi
	RGBSpectrum sensor_linear_srgb_radiance = RGBSpectrum(fs.weight) * camera_weight *
		Integrator::Li(shader_data, primary_ray, sampler, &visible_surface, &debugdata);
	sensor_linear_srgb_radiance = sensor_linear_srgb_radiance.clampOutput();

	GBuffer gbuffer = GBuffer(visible_surface);
	shader_data.gbuffer_texture.textureWriteUV(gbuffer.packGBuffer(), shading_job.uv_coord);
	VBuffer vbuffer = VBuffer(visible_surface);
	vbuffer.velocity = computeVelocity(shader_data, visible_surface);
	shader_data.vbuffer_texture.textureWriteUV(vbuffer.packVBuffer(), shading_job.uv_coord);

	float alpha = 1.0f;
	if (shader_data.renderer_settings.integrator_use_temporal_accumulation) {
		float4 temporal_color_data = reprojectAccumulate(shader_data,
			make_float2(shading_job.pixel_coord), visible_surface, vbuffer.velocity, sensor_linear_srgb_radiance);
		sensor_linear_srgb_radiance = RGBSpectrum(temporal_color_data);
		alpha = temporal_color_data.w;
	}
	else {
		//Monte-Carlo estimation; static accumulation
		sensor_linear_srgb_radiance = Integrator::addSample(shader_data, shading_job.pixel_coord, sensor_linear_srgb_radiance);
	}

	float4 frag_color = make_float4(sensor_linear_srgb_radiance.toFloat3(), alpha);

	/*
	float scale = centerMeteringWeight(frame_res, shading_job.pixel_coord, 1.0f);
	scale = ceilf(scale);
	if (!scale) frag_color += make_float4(1.0f) * length(make_float3(frag_color));

	float3 gas_heat_map = (make_float3(0, 1, 0) * visible_surface.blas_hits * 0.02f) +
		(make_float3(0, 0, 1) * visible_surface.tlas_hits * 0.05f);

	GBuffer prev = GBuffer::unpackGBuffer(shader_data.prev_gbuffer_texture.textureReadNearest(make_float2(shading_job.pixel_coord)));
	vbuffer.velocity *= 10.0f;
	shader_data.debug_texture.textureWriteUV(make_float4(vbuffer.velocity.x, vbuffer.velocity.y, 0.0f, 1.0f), shading_job.uv_coord);
	*/
	shader_data.main_texture.textureWriteUV(frag_color, shading_job.uv_coord);
}

__global__ void modulateSamples(const KittlesPT::ShaderData shader_data)
{
	using namespace KittlesPT;

	ShadingJob shading_job = getShadingJob(shader_data.frame_resolution);

	if (shading_job.invalid) {
		return;
	}

	RGBSpectrum sensor_linear_srgb_radiance = RGBSpectrum(shader_data.main_texture.textureReadNearestUV(shading_job.uv_coord));

	GBuffer gbuffer = GBuffer::unpackGBuffer(shader_data.gbuffer_texture.textureReadNearest(make_float2(shading_job.pixel_coord)));
	//Simulate 1st interaction spectral reflectance
	sensor_linear_srgb_radiance *= RGBSpectrum(gbuffer.albedo);//Modulate

	shader_data.main_texture.textureWrite(make_float4(sensor_linear_srgb_radiance.toFloat3(), 1), shading_job.pixel_coord);
}

//Conversion from radiance to screen pixels (task 3)
__global__ void computePostProcess(const KittlesPT::ShaderData shader_data)
{
	using namespace KittlesPT;

	ShadingJob shading_job = getShadingJob(shader_data.frame_resolution);

	if (shading_job.invalid) {
		return;
	}

	RGBSpectrum sensor_linear_srgb_radiance = RGBSpectrum(shader_data.main_texture.textureReadNearestUV(shading_job.uv_coord));

	if (shader_data.renderer_settings.bloom_generate_bloom) {
		RGBSpectrum veiling_linear_srgb_radiance = RGBSpectrum(shader_data.bloom_texture.textureReadNearestUV(shading_job.uv_coord));
		sensor_linear_srgb_radiance = lerp(sensor_linear_srgb_radiance, veiling_linear_srgb_radiance, shader_data.renderer_settings.bloom_final_blend);
	}

	{
		float3 Yxy = sensor_linear_srgb_radiance.toYxy();
		Yxy.x *= shader_data.scene_camera.getFilm().luminance_exposure_scalar;//scale scene luminance; TODO: stupid API
		sensor_linear_srgb_radiance = RGBSpectrum::fromYxy(Yxy);
	}

	float3 frag_color = shader_data.scene_camera.getFilm().computeNormalizedNonLinearSRGB(sensor_linear_srgb_radiance);
	//frag_color = make_float3(shader_data.gbuffer_texture.textureReadNearest(make_float2(shading_job.pixel_coord)));
	//non-linear srgb target; expects gamma correction
	shader_data.backbuffer_texture.textureWriteUV(make_float4(frag_color, 1.0f), shading_job.uv_coord);
}

__global__ void computeEffects(const KittlesPT::ShaderData shader_data)
{
	using namespace KittlesPT;

	ShadingJob shading_job = getShadingJob(shader_data.frame_resolution);

	if (shading_job.invalid) {
		return;
	}
	float2 uv = getFishEyeUV(make_float2(shading_job.pixel_coord), make_float2(shading_job.work_size));
	float2 offset = powf(fabs(shading_job.uv_coord - make_float2(0.5)) * 2.0f, 2.0f) * 0.01f;
	float r = RGBSpectrum(shader_data.backbuffer_texture.textureReadNearestUV(uv + offset)).r;
	float g = RGBSpectrum(shader_data.backbuffer_texture.textureReadNearestUV(uv)).g;
	float b = RGBSpectrum(shader_data.backbuffer_texture.textureReadNearestUV(uv - offset)).b;

	RGBSpectrum sensor_radiance(r, g, b);

	shader_data.main_texture.textureWriteUV(make_float4(sensor_radiance.toFloat3(), 1.0f), shading_job.uv_coord);
}