#pragma once
#include "triangle.cuh"
#include <vector_types.h>
#include <cuda/std/span>

namespace KittlesPT
{
	class Sphere
	{
	public:
		__host__ Sphere(float radius_, float3 pos, int material_id, int light_id) :
			radius(radius_), world_position(pos), material_id(material_id), light_id(light_id) {};

		__device__ Intersection intersect(const Ray& ray, float tmax) const
		{
			float3 oc = ray.getOrigin() - world_position;
			float b = dot(ray.getDirection(), oc); // Linear term (no need to multiply by 2)
			float c = dot(oc, oc) - radius * radius; // Avoid Sqr() call for simplicity
			float discriminant = b * b - c; // Simplified discriminant (divided quadratic equation by 4)

			Intersection intr;
			intr.distance = -1.0f; // Default: no intersection

			if (discriminant >= 0.0f)
			{
				float sqrtD = sqrtf(discriminant);
				float t0 = -b - sqrtD; // Smaller root
				float t1 = -b + sqrtD; // Larger root

				// Select the nearest valid root
				if (t0 > 0.0f && t0 < tmax)
				{
					intr.distance = t0;
				}
				else if (t1 > 0.0f && t1 < tmax)
				{
					intr.distance = t1;
				}
			}

			return intr;
		}

		__device__ ShapeSample sample(float2 u2) const {
			ShapeSample ss;
			ss.wpos = world_position + (radius * sampleUniformSphere(u2));
			ss.wgnorm = normalize(ss.wpos - world_position);
			ss.pdf = 1.0f / getArea();
			return ss;
		}

		//TODO: UV is unset!
		__device__ ShapeSample sample(float2 u2, ShapeSampleContext ctx) const
		{
			float sinThetaMax = radius / distance(ctx.wpos, world_position);
			float sin2ThetaMax = Sqr(sinThetaMax);
			float cosThetaMax = sqrtf(1 - sin2ThetaMax);
			float oneMinusCosThetaMax = 1 - cosThetaMax;

			float cosTheta = (cosThetaMax - 1) * u2.x + 1;
			float sin2Theta = 1 - Sqr(cosTheta);

			float cosAlpha = sin2Theta / sinThetaMax +
				cosTheta * sqrtf(1 - sin2Theta / Sqr(sinThetaMax));
			float sinAlpha = sqrtf(1 - Sqr(cosAlpha));

			float phi = u2.y * 2 * Constants::PI;
			float3 w = toSphericalDirection(sinAlpha, cosAlpha, phi);

			Mat3 samplingFrame = generateONBFrisvad(normalize(world_position - ctx.wpos));
			float3 n = samplingFrame.inverse() * (-w);
			cuda::std::swap(n.y, n.z);

			float3 p = world_position + radius * n;
			float pdf = 1.0f / (2.0f * Constants::PI * oneMinusCosThetaMax);

			return ShapeSample(p, n, pdf, {});
		}

		__device__ float getArea() const
		{
			return 4.0f * Constants::PI * Sqr(radius);
		}

		__device__ float getProjectedArea() const
		{
			return Constants::PI * Sqr(radius);
		}

	public:
		int material_id = -1;
		int light_id = -1;
		float3 world_position{ 0.0f,0.0f,0.0f };
		float radius = 1.0f;
	};
}/*KittlesPT*/