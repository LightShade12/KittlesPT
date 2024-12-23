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
		if (discriminant < 0)
		{
			intr.distance = -1.0;
		}
		else
		{
			float sqrtD = sqrtf(discriminant);
			float t1 = (h - sqrtD) / a; // Smaller root
			float t2 = (h + sqrtD) / a; // Larger root

			// Choose the closest positive root
			if (t1 > 0.0f)
			{
				intr.distance = t1;
			}
			else if (t2 > 0.0f)
			{
				intr.distance = t2;
			}
			else
			{
				intr.distance = -1.0f; // Both roots are negative
			}
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
			mat.roughness,
			mat.transmission,
			mat.ior,
			backface);
		return bsdf;
	}

	__device__ Ray SurfaceInteraction::spawnRay(float3 wi, int scatter_flags)
	{
		float3 ray_orig;
		if (scatter_flags & BSDFSample::Scatter::Transmitted)
		{
			ray_orig = world_position;
		}
		else
		{
			ray_orig = world_position + (world_geometric_normal * Constants::HIT_EPSILON);
		}
		return Ray(ray_orig, wi);
	}
}