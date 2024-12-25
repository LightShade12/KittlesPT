#pragma once

#include "ray.cuh"
#include "bsdf.cuh"
#include "samplers.cuh"
#include "maths/linear_algebra.cuh"

namespace KittlesPT
{
	struct GlobalShaderData;

	struct SurfaceInteraction
	{
		float distance = -1;
		float3 world_position;
		float3 world_geometric_normal;
		int material_id = -1;
		bool backface = false;

		__device__ RGBSpectrum Le(const GlobalShaderData& shader_data, const Ray& ray) const;

		__device__ BSDF getBSDF(const GlobalShaderData& shader_data) const;

		__device__ Ray spawnRay(float3 wi, int scatter_flags) const;

		__device__ Ray spawnRayTo(float3 target) const;
	};

	struct Intersection
	{
		__device__ bool operator ! ();
		float distance = -1;
		int instance_id = -1;

		//closest hit shader
		__device__ SurfaceInteraction getSurfaceInteraction(const GlobalShaderData& shader_data, const Ray& ray);
	};

	struct ShapeSample
	{
		float3 geo_w_normal;
		float3 point;
		float pdf = 0;
	};

	class Sphere
	{
	public:
		__device__ __host__ Sphere(float radius_, float3 pos, int material_id)//TODO: make it host only
			:radius(radius_), world_position(pos), material_id(material_id) {};

		__device__ Intersection intersect(const Ray& ray, float tmax) const;

		__device__ ShapeSample sample(float2 u2) const
		{
			ShapeSample ss;
			ss.point = world_position + (radius * sampleUniformSphere(u2));
			ss.geo_w_normal = normalize(ss.point = world_position);
			ss.pdf = 1.0f / getArea();
			return ss;
		}
		__host__ __device__ float getArea() const
		{
			return 4.0f * Constants::PI * Sqr(radius);
		}
		__host__ __device__ float getProjectedArea() const
		{
			return Constants::PI * Sqr(radius);
		}

		int material_id = -1;
		float3 world_position;
		float radius = 1;
	};
}