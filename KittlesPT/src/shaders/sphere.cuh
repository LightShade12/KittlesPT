#pragma once

#include "light.cuh"
#define __CUDACC__
#include <vector_types.h>

namespace KittlesPT
{
	struct GlobalShaderData;
	class RGBSpectrum;
	class Ray;
	class BSDF;

	struct SurfaceInteraction
	{
		__device__ RGBSpectrum Le(const GlobalShaderData& shader_data, const Ray& ray) const;

		__device__ BSDF getBSDF(const GlobalShaderData& shader_data) const;

		__device__ Ray spawnRay(float3 wi, int scatter_flags) const;

		__device__ Ray spawnRayTo(float3 target) const;

		//--------------------------------------------------
		float distance = -1;
		float2 uv;
		float3 world_position;
		float3 world_geometric_normal;
		int material_id = -1;
		bool backface = false;
		const Light* light = nullptr;
	};

	struct Intersection
	{
		//closest hit shader
		__device__ SurfaceInteraction getSurfaceInteraction(const GlobalShaderData& shader_data, const Ray& ray);

		__device__ bool operator ! ();

		//--------------------------------------------------------
		float distance = -1;
		int instance_id = -1;
	};

	//=======================================================================================================================

	struct ShapeSample
	{
		ShapeSample() = default;

		__device__ ShapeSample(float3 wpos, float3 gwnorm, float pdf) :
			wpos(wpos), wgnorm(gwnorm), pdf(pdf) {};

		float3 wgnorm{};
		float3 wpos{};
		float pdf = 0;
	};

	struct ShapeSampleContext
	{
		__device__ ShapeSampleContext(const SurfaceInteraction& surf) :
			wpos(surf.world_position), wgnorm(surf.world_geometric_normal)
		{};

		__device__ ShapeSampleContext(const LightSampleContext& ctx) :
			wpos(ctx.w_pos), wgnorm(ctx.wgnorm)
		{};

		float3 wpos;
		float3 wgnorm;
	};

	class Sphere
	{
	public:
		__device__ __host__ Sphere(float radius_, float3 pos, int material_id, int light_id)//TODO: make it host only
			:radius(radius_), world_position(pos), material_id(material_id), light_id(light_id) {};

		__device__ Intersection intersect(const Ray& ray, float tmax) const;

		__device__ ShapeSample sample(float2 u2) const;

		__device__ ShapeSample sample(float2 u2, ShapeSampleContext ctx) const;

		__host__ __device__ float getArea() const;

		__host__ __device__ float getProjectedArea() const;

	public:
		int material_id = -1;
		int light_id = -1;
		float3 world_position;
		float radius = 1;
	};
}