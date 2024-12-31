#include "sphere.cuh"
#include "bsdf.cuh"

#include "samplers.cuh"
#include "ray.cuh"
#include "maths/linear_algebra.cuh"
#include "containers.cuh"
#include "material.cuh"

#include <cuda/std/span>

namespace KittlesPT
{
	__device__ Intersection Sphere::intersect(const Ray& ray, float tmax) const
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
			if (t1 > 0.0f && t1 < tmax)
			{
				intr.distance = t1;
			}
			else if (t2 > 0.0f && t2 < tmax)
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

	__device__ ShapeSample Sphere::sample(float2 u2) const
	{
		ShapeSample ss;
		ss.wpos = world_position + (radius * sampleUniformSphere(u2));
		ss.wgnorm = normalize(ss.wpos - world_position);
		ss.pdf = 1.0f / getArea();
		return ss;
	}

	__device__ ShapeSample Sphere::sample(float2 u2, ShapeSampleContext ctx) const
	{
		float sinThetaMax = radius / length(ctx.wpos - world_position);
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

		return ShapeSample(p, n, pdf);
	}

	__host__ __device__ float Sphere::getArea() const
	{
		return 4.0f * Constants::PI * Sqr(radius);
	}

	__host__ __device__ float Sphere::getProjectedArea() const
	{
		return Constants::PI * Sqr(radius);
	}

	//=========================================================================================

	__device__ bool Intersection::operator!()
	{
		return (instance_id < 0);
	}

	__device__ SurfaceInteraction Intersection::getSurfaceInteraction(const GlobalShaderData& shader_data, const Ray& ray)
	{
		SurfaceInteraction surfintr;
		const Sphere& sphere = shader_data.geometry_buffer.data[instance_id];
		float3 wo = -ray.getDirection();

		surfintr.distance = distance;
		surfintr.material_id = sphere.material_id;

		surfintr.world_position = ray.getPointAt(distance);
		surfintr.world_geometric_normal = normalize(surfintr.world_position - sphere.world_position);

		if (dot(surfintr.world_geometric_normal, wo) < 0)
		{
			surfintr.world_geometric_normal *= -1.0f;
			surfintr.backface = true;
		}

		if (sphere.light_id >= 0) {
			surfintr.light = &(shader_data.lights_buffer.data[sphere.light_id]);
		}

		float3 p = (surfintr.world_position - sphere.world_position) / sphere.radius;
		float theta = acosf(-p.y);
		float phi = atan2(-p.z, p.x) + Constants::PI;

		surfintr.uv.x = phi / (2.0f * Constants::PI);
		surfintr.uv.y = theta / Constants::PI;

		return surfintr;
	}

	//=========================================================================================

	__device__ RGBSpectrum SurfaceInteraction::Le(const GlobalShaderData& shader_data, const Ray& ray) const
	{
		RGBSpectrum emission(0);
		if (!light)
		{
			return emission;
		}

		const Material& mat = shader_data.materials_buffer.data[material_id];
		emission = RGBSpectrum(mat.emissive_factor * mat.emission_scale);
		return emission;
	}

	__device__ BSDF SurfaceInteraction::getBSDF(const GlobalShaderData& shader_data) const
	{
		const Material& mat = shader_data.materials_buffer.data[material_id];

		float3 albedo = mat.albedo;

		if (mat.albedo_texture_id >= 0)
		{
			albedo = shader_data.texture_buffer.data[mat.albedo_texture_id].evaluate(shader_data,
				*this).toFloat3();
			//albedo = lerp(make_float3(1, 0, 0), make_float3(0, 1, 0), uv.y);
			//albedo = make_float3(uv.x, 0, uv.y);
		}

		BSDF bsdf = BSDF(generateONBFrisvad(world_geometric_normal),
			albedo,
			mat.metallicity,
			mat.roughness,
			mat.transmission,
			mat.ior,
			backface);
		return bsdf;
	}

	__device__ Ray SurfaceInteraction::spawnRay(float3 wi, int scatter_flags) const
	{
		float3 ray_orig;
		if (scatter_flags & BSDFSample::Scatter::Transmitted)
		{
			ray_orig = world_position - (world_geometric_normal * Constants::HIT_EPSILON);
		}
		else
		{
			ray_orig = world_position + (world_geometric_normal * Constants::HIT_EPSILON);
		}
		return Ray(ray_orig, wi);
	}
	__device__ Ray SurfaceInteraction::spawnRayTo(float3 target) const
	{
		float3 ray_orig = world_position + (world_geometric_normal * Constants::HIT_EPSILON);
		return Ray(ray_orig, normalize(target - ray_orig));
	}
}