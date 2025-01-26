#pragma once
#include "maths/constants.cuh"
#include "maths/linear_algebra.cuh"
#include "ray.cuh"
#include "interaction.cuh"
#include "samplers.cuh"
#include "light.cuh"

namespace KittlesPT
{
	struct GlobalShaderData;
	class RGBSpectrum;
	class BSDF;
	struct SurfaceInteraction;

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
		{}

		__device__ ShapeSampleContext(const LightSampleContext& ctx) :
			wpos(ctx.w_pos), wgnorm(ctx.wgnorm)
		{}

		float3 wpos;
		float3 wgnorm;
	};

	struct Vertex
	{
		Vertex(float3 p, float3 n, float2 tex_coords) :
			position(p), normal(n), tex_coords(tex_coords) {}
		/*Vertex(glm::vec3 p, glm::vec3 n, glm::vec2 tex_coords) :
			position(make_float3(p.x, p.y, p.z)),
			normal(make_float3(n.x, n.y, n.z)),
			tex_coords(make_float2(tex_coords.x, tex_coords.y))
		{}*/
		float3 position;
		float3 normal;
		float2 tex_coords;
	};

	class Triangle
	{
	public:
		__host__ Triangle(Vertex v0, Vertex v1, Vertex v2, int material_id, int light_id) :
			vertex0(v0), vertex1(v1), vertex2(v2), material_id(material_id), light_id(light_id)
		{
			//geometric normal construction
			float3 edge0 = vertex1.position - vertex0.position;
			float3 edge1 = vertex2.position - vertex0.position;
			float3 geo_norm = cross(edge0, edge1);

			float3 avg_vertex_normal = (vertex0.normal + vertex1.normal + vertex2.normal) / 3.f;

			float shn_gn_dot = dot(geo_norm, avg_vertex_normal);
			geometric_normal = (shn_gn_dot < 0.0f) ? -geo_norm : geo_norm;
		};

		__device__ void intersect(const Ray& ray, float tmax, Intersection* intr) const
		{
			intr->distance = -1.0f;

			float3 v0v1 = vertex1.position - vertex0.position;
			float3 v0v2 = vertex2.position - vertex0.position;

			float3 pvec = cross(ray.getDirection(), v0v2);

			float det = dot(v0v1, pvec);
			if (det > -Constants::TRIANGLE_INTERSECTION_EPSILON && det < Constants::TRIANGLE_INTERSECTION_EPSILON) {
				return; //parallel
			}
			float invDet = 1.0f / det;

			float3 tvec = ray.getOrigin() - vertex0.position;
			float u = invDet * dot(tvec, pvec);
			if (u < 0.0f || u > 1.0f) {
				return;
			}

			float3 qvec = cross(tvec, v0v1);
			float v = invDet * dot(ray.getDirection(), qvec);
			if (v < 0.0f || u + v > 1.0f) {
				return;
			}

			float t = invDet * dot(v0v2, qvec);
			// ray intersection
			if (t > Constants::TRIANGLE_INTERSECTION_EPSILON && t < tmax) {
				intr->distance = t;
				intr->bary_coords = make_float3(1.0f - u - v, u, v);

				return;
			}

			return;
		}

		__device__ ShapeSample sample(float2 u2, ShapeSampleContext ctx) const {
			float3 p0 = vertex0.position, p1 = vertex1.position, p2 = vertex2.position;
			float3 bary = sampleUniformTriangle(u2);
			float3 p = p0 * bary.x + p1 * bary.y + p2 * bary.z;
			float pdf = 1.0f / getArea();
			pdf *= Sqr(length(p - ctx.wpos));
			pdf /= AbsDot(normalize(p - ctx.wpos), geometric_normal);
			return ShapeSample(p, geometric_normal, pdf);
		};

		__device__ float getArea() const {
			return 0.5f * length(cross(vertex1.position - vertex0.position, vertex2.position - vertex0.position));
		};

	public:
		Vertex vertex0, vertex1, vertex2;
		float3 geometric_normal{ 0.0f,0.0f,0.0f };
		int material_id = -1;
		int light_id = -1;
		int primitive_id = -1;
	};
}/*KittlesPT*/