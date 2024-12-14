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

		int MAX_RAY_DEPTH = 3;
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

	frag_color = sensorL(shader_data, primary_ray, sampler);

	float4 old_col = shader_data.main_texture.textureReadNearest(pixel_coord);

	frag_color = lerp(make_float3(old_col), frag_color, 0.1);

	shader_data.main_texture.textureWrite(make_float4(frag_color, 1), pixel_coord);
}