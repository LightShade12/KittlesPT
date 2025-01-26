#include "interaction.cuh"
#include "containers.cuh"
#include "ray.cuh"
#include "bsdf.cuh"
#include "maths/constants.cuh"

namespace KittlesPT
{
	__device__ SurfaceInteraction Intersection::getSurfaceInteraction(const GlobalShaderData& shader_data, const Ray& ray)
	{
		SurfaceInteraction surfintr;
		const Triangle& tri = shader_data.triangles_buffer.data[primitive_id];
		float3 wo = -ray.getDirection();

		surfintr.distance = distance;
		surfintr.material_id = tri.material_id;

		surfintr.world_position = ray.getPointAt(distance);
		//surfintr.world_position = (bary_coords.x * tri.vertex0.position) + (bary_coords.y * tri.vertex1.position) + (bary_coords.z * tri.vertex2.position);

		surfintr.world_geometric_normal = tri.geometric_normal;

		if (dot(surfintr.world_geometric_normal, wo) < 0)
		{
			surfintr.world_geometric_normal *= -1.0f;
			surfintr.backface = true;
		}

		if (tri.light_id >= 0) {
			surfintr.arealight = &(shader_data.lights_buffer.data[tri.light_id]);
		}

		surfintr.uv = (bary_coords.x * tri.vertex0.tex_coords) + (bary_coords.y * tri.vertex1.tex_coords) + (bary_coords.z * tri.vertex2.tex_coords);

		/*
		float3 p = (surfintr.world_position - tri.world_position) / tri.radius;
		p = surfintr.world_geometric_normal;
		float theta = acosf(-p.y);
		float phi = atan2(-p.z, p.x) + Constants::PI;
		surfintr.uv.x = phi / (2.0f * Constants::PI);
		surfintr.uv.y = theta / Constants::PI;
		*/

		return surfintr;
	}

	//=========================================================================================

	__device__ RGBSpectrum SurfaceInteraction::Le(const GlobalShaderData& shader_data, const Ray& ray) const
	{
		return(arealight) ? arealight->L(shader_data, uv) : RGBSpectrum(0.0f);
	}

	__device__ BSDF SurfaceInteraction::getBSDF(const GlobalShaderData& shader_data) const
	{
		const Material& mat = shader_data.materials_buffer.data[material_id];

		BSDF bsdf = mat.getBSDF(shader_data, MaterialEvalContext(*this));

		return bsdf;
	}

	__device__ void SurfaceInteraction::skipInteraction(Ray* ray)
	{
		float3 offset = world_geometric_normal * Constants::HIT_EPSILON;
		if (dot(ray->getDirection(), world_geometric_normal) < 0) {
			offset *= -1.0f;
		}
		float3 orig = world_position + offset;
		*ray = Ray(orig, ray->getDirection());
	}

	__device__ Ray SurfaceInteraction::spawnRay(float3 wi, int scatter_flags) const
	{
		float3 ray_orig;
		if (scatter_flags & BSDFSample::Transmitted) {
			ray_orig = world_position - (world_geometric_normal * Constants::HIT_EPSILON);
		}
		else {
			ray_orig = world_position + (world_geometric_normal * Constants::HIT_EPSILON);
		}
		return Ray(ray_orig, wi);
	}

	__device__ Ray SurfaceInteraction::spawnRayTo(float3 target) const
	{
		float3 ray_orig;
		if (backface) {
			ray_orig = world_position - (world_geometric_normal * Constants::HIT_EPSILON);
		}
		else {
			ray_orig = world_position + (world_geometric_normal * Constants::HIT_EPSILON);
		}
		return Ray(ray_orig, normalize(target - ray_orig));
	}
}/*KittlesPT*/