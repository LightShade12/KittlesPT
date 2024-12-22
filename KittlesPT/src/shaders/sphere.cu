#include "sphere.cuh"
#include "containers.cuh"
#include "material.cuh"

namespace KittlesPT
{
	__device__ Intersection Sphere::intersect(const Ray& ray) const
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

	__device__ bool Intersection::operator!()
	{
		return (instance_id < 0);
	}

	__device__ BSDF SurfaceInteraction::getBSDF(const GlobalShaderData& shader_data)
	{
		const Material& mat = shader_data.materials_buffer.data[material_id];
		BSDF bsdf = BSDF(generateONBFrisvad(world_geometric_normal),
			mat.albedo,
			mat.metallicity,
			mat.roughness);
		return bsdf;
	}
	__device__ Ray SurfaceInteraction::spawnRay(float3 wi)
	{
		float3 ray_orig = world_position + (world_geometric_normal * Constants::HIT_EPSILON);
		return Ray(ray_orig, wi);
	}
}