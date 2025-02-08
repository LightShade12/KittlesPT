#pragma once
#include "containers.cuh"
#include "color.cuh"
#include "containers.cuh"
#include "interaction.cuh"
#include "sphere.cuh"
#include "bsdf.cuh"
#include "samplers.cuh"
#include "atmosphere.cuh"
#include "light_sampler.cuh"
#include "ray.cuh"
#include "maths/sampling.cuh"
#include "color.cuh"
#include "packing.cuh"
#include "blas.cuh"

#include <cuda.h>
#include <vector_types.h>

//TODO:use *_t types

//#define INTERSECT_DEBUG

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

	namespace Integrator
	{
		//will implicitly use GAS in ShaderData
		inline __device__ Intersection intersect(const ShaderData& shader_data, const Ray& ray, float tmin, float tmax, DebugData& dbg)
		{
			return shader_data.top_level_acceleration_structure.intersect(shader_data, ray, tmin, tmax, dbg);

#ifdef INTERSECT_DEBUG
			return shader_data.blas_buffer.data[0].intersect(shader_data, ray, tmin, tmax, dbg);

			Intersection closest;
			Intersection intr;

			for (int32_t instance_id = 0; instance_id < shader_data.meshes_buffer.num; instance_id++)
			{
				const TriangleMesh& mesh = shader_data.meshes_buffer.data[instance_id];
				Ray transformed_ray = ray.transform(mesh.inv_model_matrix);

				for (int32_t primitive_id = mesh.prim_offset; primitive_id < mesh.prim_offset + mesh.prim_count; primitive_id++)
				{
					const Triangle& tri = shader_data.triangles_buffer.data[primitive_id];
					tri.intersect(transformed_ray, tmin, tmax, &intr);
					//TODO: can just store and use pmin of tmax,INF
					if (intr.distance < INFINITY && intr.distance < tmax && intr.distance > tmin && intr.distance < closest.distance)
					{
						closest = intr;
						closest.primitive_id = primitive_id;
						closest.instance_id = instance_id;
					}
				}
			}
			return closest;
#endif//INTERSECT_DEBUG
		}

		inline __device__ bool intersectShadow(const ShaderData& shader_data, const Ray& ray, float tmin, float tmax, DebugData& dbg)
		{
			return shader_data.top_level_acceleration_structure.intersectP(shader_data, ray, tmin, tmax);

#ifdef INTERSECT_DEBUG
			return shader_data.blas_buffer.data[0].intersectP(shader_data, ray, tmin, tmax);

			Intersection intr;
			for (int32_t instance_id = 0; instance_id < shader_data.meshes_buffer.num; instance_id++)
			{
				const TriangleMesh& mesh = shader_data.meshes_buffer.data[instance_id];
				Ray object_ray = ray;
				//TODO: ensure normality
				object_ray.setDirection(make_float3(mesh.inv_model_matrix * make_float4(object_ray.getDirection(), 0)));
				object_ray.setOrigin(make_float3(mesh.inv_model_matrix * make_float4(object_ray.getOrigin(), 1)));

				for (int32_t primitive_id = mesh.prim_offset; primitive_id < mesh.prim_offset + mesh.prim_count; primitive_id++)
				{
					const Triangle& tri = shader_data.triangles_buffer.data[primitive_id];
					tri.intersect(object_ray, tmin, tmax, &intr);
					if (intr.distance < INFINITY && intr.distance < tmax && intr.distance>tmin)
					{
						return true;
					}
				}
			}
			return false;
#endif//INTERSECT_DEBUG
		}

		inline __device__ bool Unoccluded(const ShaderData& shader_data, const SurfaceInteraction& surface, float3 target)
		{
			constexpr float SHADOWRAY_EPSILON = 0.11f;//TODO: put this in a constants file
			Ray shadow_ray = surface.spawnRayTo(target);
			float tmax = distance(target, shadow_ray.getOrigin()) - SHADOWRAY_EPSILON;
			DebugData dummy;
			return (!intersectShadow(shader_data, shadow_ray, 0.0f, tmax, dummy));
		}

		//----------------------------------------------------------------

		inline __device__ RGBSpectrum LeSun(const ShaderData& shader_data, const Ray& ray, const Atmosphere& atmosphere)
		{
			float3 atmosphere_observer_position = make_float3(0.0f, atmosphere.getEarthRadiusMeters() + 1.0f, 0.0f);

			float min_similarity_threshold = cosf(shader_data.procedural_environment_data.sun_angular_diameter_rad / 2.0f);
			float similarity = dot(ray.getDirection(), atmosphere.getSunDirection());
			bool sun_surface = similarity > min_similarity_threshold;//step
			//TODO:fucked sun emission
			float sun_nits = shader_data.procedural_environment_data.sun_emission_nits;
			return RGBSpectrum(sun_nits * ::powf(1.0f / sun_nits, !sun_surface));//branchless
		}

		inline __device__ RGBSpectrum sampleLdSun(const ShaderData& shader_data, const Ray& ray, const BSDF& bsdf,
			const SurfaceInteraction& surface, const Atmosphere& atmosphere, IndependentSampler& sampler)
		{
			RGBSpectrum Ld(0.0f);
			const float3& sun_direction = atmosphere.getSunDirection();
			float3 sun_position = sun_direction * SUN_PHYSICAL_DISTANCE_METERS;
			float sun_radius_meters = angularDiameterToPhysicalDiameter(shader_data.procedural_environment_data.sun_angular_diameter_rad,
				SUN_PHYSICAL_DISTANCE_METERS) / 2.0f;
			float3 sample_offset = make_float3(sampler.get2D() * 2.0f - 1.0f, sampler.get1D() * 2.0f - 1.0f);//TODO: fix sampling

			float3 target = sun_position + (sample_offset * sun_radius_meters);

			float3 wo = -ray.getDirection();
			float3 atmosphere_observer_position = make_float3(0.0f, atmosphere.getEarthRadiusMeters() + 1.0f, 0.0f);

			RGBSpectrum fcos = bsdf.f(wo, sun_direction) *
				fmaxf(0.0f, dot(sun_direction, ((surface.backface) ? -1.0f : 1.0f) * surface.world_geometric_normal));

			if (!fcos) {
				return Ld;
			}

			RGBSpectrum sun_radiance = atmosphere.sampleLe(Ray(atmosphere_observer_position, sun_direction));

			if (!sun_radiance) {
				return Ld;
			}

			if (!Unoccluded(shader_data, surface, target)) {
				return Ld;
			};

			float sun_area = Constants::PI * Sqr(sun_radius_meters);
			float3 sun_surf_nrm = normalize(target - sun_position), wi = normalize(target - surface.world_position);
			float theta_sun = AbsDot(sun_surf_nrm, -wi);//cosine
			float pdf = (1.0f / sun_area) / (theta_sun / Sqr(SUN_PHYSICAL_DISTANCE_METERS));
			Ld = (fcos * sun_radiance * shader_data.procedural_environment_data.sun_emission_nits) / pdf;

			return Ld;
		}

		inline __device__ RGBSpectrum sampleLd(const ShaderData& shader_data, const Ray& ray, const BSDF& bsdf,
			const SurfaceInteraction& surface, const UniformLightSampler& light_sampler, IndependentSampler& sampler)
		{
			RGBSpectrum Ld(0.0f);

			SampledLight sampled_light = light_sampler.sample(sampler.get1D());
			//empty buffer
			if (!sampled_light) {
				return Ld;
			}

			LightLiSample ls = sampled_light.light->sampleLi(shader_data, LightSampleContext(surface), sampler.get2D());
			if (!ls) {
				return Ld;
			}

			float3 wi = ls.wi;
			float3 wo = -ray.getDirection();
			RGBSpectrum fcos = bsdf.f(wo, wi) *
				fmaxf(0.0f, dot(wi, ((surface.backface) ? -1.0f : 1.0f) * surface.world_geometric_normal));

			if (!fcos) {
				return Ld;
			}
			if (!Unoccluded(shader_data, surface, ls.wpos_light)) {
				return Ld;
			}
			float p_l = (sampled_light.probability * ls.pdf);
			float p_b = bsdf.pdf(wo, wi);
			float w_l = powerHeuristic(1, p_l, 1, p_b);
			Ld = (ls.L * fcos * w_l) / p_l;
			return Ld;
		}

		inline __device__ float3 getSunDirection(const ShaderData& shader_data)
		{
			float phi = shader_data.procedural_environment_data.sun_phi_rad, theta = shader_data.procedural_environment_data.sun_theta_rad;
			return normalize(make_float3(
				cosf(phi) * cosf(theta),
				sinf(theta),
				sinf(phi) * cosf(theta)
			));
		}

		inline __device__ RGBSpectrum Li(const ShaderData& shader_data, const Ray& ray_in, IndependentSampler& sampler, GBuffer* visible_surface)
		{
			//114fps
			const int32_t max_ray_depth = shader_data.renderer_settings.max_bounce_depth;

			//L and beta
			RGBSpectrum light(0.0f), throughput(1.0f);
			bool specular_bounce = false, any_non_specular_bounces = false;
			float eta_scale = 1.0;
			int32_t depth = 0;

			float p_b = 1.0f;
			LightSampleContext prev_intr_ctx{};

			//TODO:store in heap
			float3 sun_direction = getSunDirection(shader_data);
			Atmosphere atmosphere(sun_direction, shader_data.procedural_environment_data.sun_emission_nits);
			float3 atmosphere_observer_position = make_float3(0, atmosphere.getEarthRadiusMeters() + 1, 0);
			UniformLightSampler light_sampler(shader_data.lights_buffer.data, shader_data.lights_buffer.num);

			Ray ray = ray_in;
			DebugData dbg;
			//112fps
			//iterate through path vertices
			while (throughput)
			{
				sampler.setSeed(sampler.getSeed() + depth); bool first_surface = (depth == 0);

				Intersection intr = intersect(shader_data, ray, 0.0f, INFINITY, dbg);
				//Sample participating media here--

				//Handle interaction with a medium; else surface scatter--
				//110fps
				if (!intr) {
					/* MISS
					* Sampling only one InfiniteLight with bsdf sampling here,
					* without any explicit sky sampling elsewhere, so no MIS used here */
					//RGBSpectrum sky_radiance = LeSun(shader_data, ray, atmosphere);
					RGBSpectrum sky_radiance = atmosphere.sampleLe(Ray(atmosphere_observer_position, ray.getDirection())) *
						LeSun(shader_data, ray, atmosphere);

					light += sky_radiance * throughput;
					break;
				}
				//79fps => 105fps(90fps when looking through atmosphere)

				SurfaceInteraction surfintr = intr.getSurfaceInteraction(shader_data, ray);

				//101fps=>105fps(kernel root inlining)

				//Sample Le from surface
				if (RGBSpectrum Le = surfintr.Le(shader_data, ray); Le) {
					float w_l = 1.0f;
					if (surfintr.arealight && !first_surface) {
						float light_pdf = light_sampler.PMF(surfintr.arealight) * surfintr.arealight->pdf_Li(prev_intr_ctx,
							LightLiSample(surfintr));
						w_l = powerHeuristic(1, p_b, 1, light_pdf);
					}
					light += Le * w_l * throughput;
				}
				//102fps

				BSDF bsdf = surfintr.getBSDF(shader_data);
				if (!bsdf) { //skip over medium boundaries
					surfintr.skipInteraction(&ray);
					continue;
				}

				if (first_surface) {
					//using texture diffuse albedo for reflectance estimate
					*visible_surface = GBuffer(bsdf.getAlbedo(), surfintr);
				}

				if (any_non_specular_bounces) {
					bsdf.regularize();
				}

				if (depth++ == max_ray_depth) {
					break;
				}

				//99fps

				if (bsdf.isNonSpecular()) {
					RGBSpectrum Ld = sampleLd(shader_data, ray, bsdf, surfintr, light_sampler, sampler);
					light += Ld * throughput;
					RGBSpectrum Ld_sun = sampleLdSun(shader_data, ray, bsdf, surfintr, atmosphere, sampler);
					light += Ld_sun * throughput;
				}

				float3 wo = -ray.getDirection();
				BSDFSample bs = bsdf.sampleF(wo, sampler.get2D(), sampler.get2D());
				if (bs.scatterTypeIs(BSDFSample::Absorbed)) {
					break;
				}

				throughput *= bs.f * AbsDot(surfintr.world_geometric_normal, bs.wi) / bs.pdf;//uses absdot for allowing refraction
				p_b = bsdf.pdf(wo, bs.wi);
				specular_bounce = bs.scatterTypeIs(BSDFSample::Specular);
				any_non_specular_bounces |= !specular_bounce;
				prev_intr_ctx = LightSampleContext(surfintr);

				ray = surfintr.spawnRay(bs.wi, bs.scatter);
				if (russianRoulette(&throughput, eta_scale, depth, sampler)) {
					break;
				}
				//90fps
				return RGBSpectrum(0);
			}
			visible_surface->blas_hits = dbg.blas_hits;
			visible_surface->tlas_hits = dbg.tlas_hits;
			return light;
		}

		//----------------------------------------------------------------
		//Monte-Carlo estimation; static accumulation
		inline __device__ RGBSpectrum addSample(const ShaderData& shader_data, int2 pixel_coord, const RGBSpectrum& radiance_sample)
		{
			RGBSpectrum accumulated_sample = RGBSpectrum(shader_data.accumulation_texture.textureReadNearest(make_float2(pixel_coord)));
			RGBSpectrum new_accumulated_sample = accumulated_sample + radiance_sample;

			shader_data.accumulation_texture.textureWrite(make_float4(new_accumulated_sample.toFloat3(), 1), pixel_coord);
			RGBSpectrum integral_estimate = new_accumulated_sample / float(shader_data.frame_index + 1);

			return integral_estimate;
		}
	}
}/*KittlesPT*/