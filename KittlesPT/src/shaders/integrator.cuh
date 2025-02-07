#pragma once
#include <cuda.h>
#include <vector_types.h>

namespace KittlesPT
{
	/*TODO:list of features below
	*
	*	-PBRT Parity
	*	-Triangles
	*	-Raymarching (Volumetrics)
	*	-Procedural Clouds
	*	-Utility code(From GLSL,HLSL .etc)
	*	-Multipsample temporal accumulation
	*	-Fix Normal maps
	*
	*	-Wavefront rendering
	*	-Mediums (Volumetric Rendering)
	*	-Stratified Sampling
	* 
	*	-Anisotropy
	*	-Specular material; specular/any_non_specular_bounces
	*	-Path regularization
	*
		- PBRT base types:
	*	- Primitive (Accelerator)
	*	- Medium
	*
	*	Performance Milestone:
	*	1> 30 fps
	*	2> 60 fps
	*	First-Person camera navigation through the 3D scene.
	*	ClearCoat(cars, plastic, polished wood, billiard balls, etc.),
	*	Translucent (skin, leaves, cloth, etc.),
	*	Subsurface w/ shiny coat (jelly beans, cherries, teeth, polished Jade, etc.)
	*	Beer-Lambert law for ray color/energy attenuation.
	*	Raytraced DOF
	*	Proper SuperSampling Integration
	*/

	struct ShaderData;
	struct Intersection;
	struct GBuffer;
	struct DebugData;
	struct SurfaceInteraction;

	class Ray;
	class BSDF;
	class Atmosphere;
	class UniformLightSampler;
	class RGBSpectrum;
	class IndependentSampler;

	namespace Integrator
	{
		//will implicitly use GAS in ShaderData
		__device__ Intersection intersect(const ShaderData& shader_data, const Ray& ray, float tmin, float tmax, DebugData& dbg);

		__device__ bool intersectShadow(const ShaderData& shader_data, const Ray& ray, float tmin, float tmax, DebugData& dbg);

		__device__ bool Unoccluded(const ShaderData& shader_data, const SurfaceInteraction& surface, float3 target);

		//----------------------------------------------------------------

		__device__ RGBSpectrum LeSun(const ShaderData& shader_data, const Ray& ray, const Atmosphere& atmosphere);

		__device__ RGBSpectrum sampleLdSun(const ShaderData& shader_data, const Ray& ray, const BSDF& bsdf,
			const SurfaceInteraction& surface, const Atmosphere& atmosphere, IndependentSampler& sampler);

		__device__ RGBSpectrum sampleLd(const ShaderData& shader_data, const Ray& ray, const BSDF& bsdf,
			const SurfaceInteraction& surface, const UniformLightSampler& light_sampler, IndependentSampler& sampler);

		__device__ RGBSpectrum Li(const ShaderData& shader_data, const Ray& ray_in, IndependentSampler& sampler, GBuffer* visible_surface);

		//----------------------------------------------------------------
		//Monte-Carlo estimation; static accumulation
		__device__ RGBSpectrum addSample(const ShaderData& shader_data, int2 pixel_coord, const RGBSpectrum& radiance_sample);
	}
}/*KittlesPT*/