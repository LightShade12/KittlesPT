#pragma once
#include <cuda.h>
#include <vector_types.h>

namespace KittlesPT
{
	/*TODO:list of features below
		*	-BBOX
		*	-BVH
		*	-Triangles
		*
		*	-Filter
		*
		*	-Wavefront rendering
		*
		*	-Utility code
		*
		*	-Anisotropy
		*	-Specular material; specular/any_non_specular_bounces
		*	-Path regularization
		*/

	struct GlobalShaderData;
	struct Intersection;
	struct GBuffer;
	struct SurfaceInteraction;

	class Ray;
	class BSDF;
	class Atmosphere;
	class LightSampler;
	class RGBSpectrum;
	class IndependentSampler;

	namespace Integrator
	{
		__device__ Intersection intersect(const GlobalShaderData& shader_data, const Ray& ray, float tmax);

		__device__ bool intersectShadow(const GlobalShaderData& shader_data, const Ray& ray, float tmax);

		__device__ bool Unoccluded(const GlobalShaderData& shader_data, const SurfaceInteraction& surface, float3 target);

		__device__ RGBSpectrum sampleLdSun(const GlobalShaderData& shader_data, const Ray& ray, float3 sun_direction, const BSDF& bsdf,
			const SurfaceInteraction& surface, const Atmosphere& atmosphere, IndependentSampler& sampler);

		__device__ RGBSpectrum sampleLd(const GlobalShaderData& shader_data, const Ray& ray, const BSDF& bsdf,
			const SurfaceInteraction& surface, const LightSampler& light_sampler, IndependentSampler& sampler);

		//russian roulette
		__device__ bool russianRoulette(RGBSpectrum& throughput, float eta_scale, int bounce_depth, IndependentSampler& sampler);

		__device__ float3 sphericalToSunDirection(float theta, float phi);

		__device__ RGBSpectrum sensorRadiance(const GlobalShaderData& shader_data, const Ray& ray_in, IndependentSampler& sampler, GBuffer* visible_surface);
	}
}