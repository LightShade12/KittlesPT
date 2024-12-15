#include "kernels.cuh"

#include "../error_check.cuh"
#include "../maths/linear_algebra.cuh"
#include "ray.cuh"
#include "bsdf.cuh"
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

	__device__ Intersection intersect(const GlobalShaderData& shader_data, const Ray& ray)
	{
		Intersection closest;
		closest.distance = INFINITY;

		for (int instance_id = 0; instance_id < shader_data.scene_buffer.num; instance_id++)
		{
			const Sphere& sphere = shader_data.scene_buffer.data[instance_id];
			Intersection intr = sphere.intersect(ray);
			if (intr.distance < closest.distance && intr.distance >= 0)
			{
				closest.distance = intr.distance;
				closest.instance_id = instance_id;
			}
		}
		return closest;
	}

	__device__ Mat3 generateOrthonormalBasis(const float3& normal)
	{
		// Choose a helper vector H that is not parallel to the normal
		float3 helper = (fabs(normal.x) > fabs(normal.z)) ? make_float3(0, 1, 0) : make_float3(1, 0, 0);

		// Compute tangent vector (orthogonal to normal)
		float3 tangent = normalize(cross(helper, normal));

		// Compute bitangent vector (orthogonal to both normal and tangent)
		float3 bitangent = cross(normal, tangent);

		return Mat3(tangent, bitangent, normal);
	}
	__device__ float3 sensorL(const GlobalShaderData& shader_data, const Ray& ray_in, IndependentSampler& sampler)
	{
		float3 light = make_float3(0);
		float3 throughput = make_float3(1);

		constexpr int MAX_RAY_DEPTH = 3;
		Ray ray = ray_in;

		for (int bounce_depth = 0; bounce_depth < MAX_RAY_DEPTH; bounce_depth++)
		{
			sampler.setSeed(sampler.getSeed() + bounce_depth);

			Intersection intr = intersect(shader_data, ray);

			if (!intr)
			{
				//miss
				float3 unit_direction = normalize(ray.getDirection());
				float a = 0.5 * (unit_direction.y + 1.0);
				float3 color = (1.0 - a) * make_float3(1.0, 1.0, 1.0) + a * make_float3(0.5, 0.7, 1.0);
				light += color * throughput;
				break;
			}

			//hit
			float3 wo = -ray.getDirection();

			SurfaceInteraction surfintr = closestHit(ray, intr, shader_data.scene_buffer.data[intr.instance_id]);

			BSDF bsdf = BSDF(generateOrthonormalBasis(surfintr.world_normal), make_float3(0.8));
			BSDFSample bs = bsdf.sampleBSDF(wo, sampler.get2D());

			float3 wi = bs.wi;
			float pdf = bs.pdf;

			float3 fcos = bs.f * dot(surfintr.world_normal, wi);
			if (!fcos) break;

			throughput *= (fcos / pdf);

			ray = Ray(surfintr.world_position + (surfintr.world_normal * Constants::HIT_EPSILON), wi);
		}

		return light;
	}

	namespace AgxMinimal
	{
		// 0: Default, 1: Golden, 2: Punchy
#define AGX_LOOK 0

//Fifth order
// Mean error^2: 3.6705141e-06
		__device__ float3 agxDefaultContrastApprox(float3 x) {
			float3 x2 = x * x;
			float3 x4 = x2 * x2;

			return +15.5 * x4 * x2
				- 40.14 * x4 * x
				+ 31.96 * x4
				- 6.868 * x2 * x
				+ 0.4298 * x2
				+ 0.1191 * x
				- 0.00232;
		}

		__device__ float3 agx_fitted(float3 col) {
			float3 val = (col);
			const Mat3 agx_mat = Mat3(
				0.842479062253094, 0.0423282422610123, 0.0423756549057051,
				0.0784335999999992, 0.878468636469772, 0.0784336,
				0.0792237451477643, 0.0791661274605434, 0.879142973793104);

			const float min_ev = -12.47393f;
			const float max_ev = 4.026069f;

			// Input transform (inset)
			val = agx_mat * val;

			// Log2 space encoding
			val = clamp(log2f(val), min_ev, max_ev);
			val = (val - min_ev) / (max_ev - min_ev);

			// Apply sigmoid function approximation
			val = agxDefaultContrastApprox(val);

			return float3(val);
		}

		__device__ float3 agx_fitted_Eotf(float3 col) {
			float3 val = (col);
			const Mat3 agx_mat_inv = Mat3(
				1.19687900512017, -0.0528968517574562, -0.0529716355144438,
				-0.0980208811401368, 1.15190312990417, -0.0980434501171241,
				-0.0990297440797205, -0.0989611768448433, 1.15107367264116);

			// Inverse input transform (outset)
			val = agx_mat_inv * val;

			// sRGB IEC 61966-2-1 2.2 Exponent Reference EOTF Display
			// NOTE: We're linearizing the output here. Comment/adjust when
			// *not* using a sRGB render target
			val = powf(val, make_float3(2.2));

			return float3(val);
		}

		__device__ float3 agxLook(float3 val)
		{
			const float3 lw = make_float3(0.2126, 0.7152, 0.0722);
			float luma = dot(val, lw);

			// Default
			float3 offset = make_float3(0.0);
			float3 slope = make_float3(1.0);
			float3 power = make_float3(1.0);
			float sat = 1.0;

#if AGX_LOOK == 1
			// Golden
			slope = make_float3(1.0, 0.9, 0.5);
			power = make_float3(0.8);
			sat = 0.8;
#elif AGX_LOOK == 2
			// Punchy
			slope = make_float3(1.0);
			power = make_float3(1.35, 1.35, 1.35);
			sat = 1.4;
#endif

			// ASC CDL
			val = powf(val * slope + offset, power);
			return luma + sat * (val - luma);
		}
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
	sampler.initPixelSeed(pixel_coord, frame_res.x, shader_data.frame_index + 1);

	Ray primary_ray = shader_data.scene_camera.generateRay(ndc_coord, frame_res);

	float3 frag_color = make_float3(ndc_coord.x, ndc_coord.y, 0.25);

	frag_color = sensorL(shader_data, primary_ray, sampler);

	shader_data.accumulation_texture.textureWrite(
		make_float4(frag_color + make_float3(shader_data.accumulation_texture.textureReadNearest(pixel_coord)), 1),
		pixel_coord);

	frag_color = make_float3(shader_data.accumulation_texture.textureReadNearest(pixel_coord)) / ((float)shader_data.frame_index + 1);

	frag_color *= 2.5f;
	frag_color = AgxMinimal::agx_fitted(frag_color);
	frag_color = AgxMinimal::agxLook(frag_color);
	frag_color = AgxMinimal::agx_fitted_Eotf(frag_color);

	shader_data.main_texture.textureWrite(make_float4(frag_color, 1), pixel_coord);
}