#include "kernels.cuh"

//#include "../renderer.hpp"
#include "../error_check.cuh"
#include "../maths/linear_algebra.cuh"
#include "ray.cuh"

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

	struct Intersection
	{
		__device__ bool operator ! ()
		{
			return (distance < 0);
		}
		float distance = -1;
	};

	struct SurfaceInteraction
	{
		float distance = -1;
		float3 world_position;
		float3 world_normal;
	};

	class Camera
	{
	public:
		__device__ Camera(float3 pos, float3 forward) :
			world_position(pos), forward_direction(forward) {};

		//generate camera rays; -1 => forawrd depth
		__device__ Ray getRay(float2 ndc_coords, int2 frame_resolution) const
		{
			float aspect_ratio = (float)frame_resolution.x / (float)frame_resolution.y;

			float camera_height = 2.0f;
			float camera_width = aspect_ratio * camera_height;

			//-1 => forward depth
			float3 raydir = make_float3(ndc_coords.x * camera_width, ndc_coords.y * camera_height, -1);
			Ray ray = Ray(world_position, normalize(raydir));
			return ray;
		};
		float3 world_position;
		float3 forward_direction;
	};

	class Sphere {
	public:
		__device__ Sphere(float radius_, float3 pos)
			:radius(radius_), world_position(pos) {};

		__device__ Intersection intersect(const Ray& ray)
		{
			float3 oc = world_position - ray.getOrigin();
			float a = Sqr(length(ray.getDirection()));
			float h = dot(ray.getDirection(), oc);
			float c = Sqr(length(oc)) - radius * radius;
			float discriminant = h * h - a * c;

			Intersection intr;
			if (discriminant < 0) {
				intr.distance = -1.0;
			}
			else {
				intr.distance = (h - sqrtf(discriminant)) / a;
			}
			return intr;
		}

		float3 world_position;
		float radius = 1;
	};

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

	Sphere sphere1(1, make_float3(0, 0, -3));
	Camera camera = Camera(make_float3(0), make_float3(0, 0, -1));

	Ray primary_ray = camera.getRay(ndc_coord, frame_res);

	Intersection hit_intr = sphere1.intersect(primary_ray);
	float3 frag_color = make_float3(ndc_coord.x, ndc_coord.y, 0.25);

	if (!hit_intr)
	{
		//miss
		frag_color = make_float3(0, 0.35, 0.45);
	}
	else
	{
		//hit
		SurfaceInteraction surfintr = closestHit(primary_ray, hit_intr, sphere1);
		frag_color = surfintr.world_normal;
	}

	shader_data.main_texture.textureWrite(make_float4(frag_color, 1), pixel_coord);
}