#include "sphere.cuh"

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
		return (distance < 0);
	}
}