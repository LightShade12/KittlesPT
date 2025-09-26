#pragma once
#include "maths/constants.cuh"
#include "maths/linear_algebra.cuh"
#include "ray.cuh"
#include "interaction.cuh"
#include "samplers.cuh"
#include "light.cuh"

namespace KittlesPT
{
	struct ShapeSample
	{
		ShapeSample() = default;

		__device__ ShapeSample(float3 wpos, float3 gwnorm, float pdf, float2 uv) :
			wpos(wpos), wgnorm(gwnorm), pdf(pdf), uv(uv) {};

		float2 uv{};
		float3 wgnorm{};
		float3 wpos{};
		float pdf = 0.0f;
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

		float3 position;
		float3 normal;
		float2 tex_coords;
	};

	class Triangle
	{
	public:
		__host__ Triangle(Vertex v0, Vertex v1, Vertex v2, int32_t material_id, int32_t light_id, int32_t mesh_id) :
			vertex0(v0), vertex1(v1), vertex2(v2), material_id(material_id), light_id(light_id), mesh_id(mesh_id)
		{
			//geometric normal construction
			float3 edge0 = vertex1.position - vertex0.position;
			float3 edge1 = vertex2.position - vertex0.position;
			float3 geo_norm = cross(edge0, edge1);

			float3 avg_vertex_normal = (vertex0.normal + vertex1.normal + vertex2.normal) / 3.f;

			float shn_gn_dot = dot(geo_norm, avg_vertex_normal);
			local_geometric_normal = (shn_gn_dot < 0.0f) ? -geo_norm : geo_norm;
			local_geometric_normal = normalize(local_geometric_normal);
		};

		__forceinline__ __device__ void intersect(const Ray& ray, float tmin, float tmax, Intersection* intr) const {
			intr->distance = INFINITY;

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
			if (t > Constants::TRIANGLE_INTERSECTION_EPSILON && t < tmax && t > tmin) {
				intr->distance = t;
				intr->bary_coords = { 1.0f - u - v, u, v };

				return;
			}

			return;
		}

		__forceinline__ __device__ ShapeSample sample(const Mat4& model, float2 u2, ShapeSampleContext ctx) const {
			float3 p0 = vertex0.position, p1 = vertex1.position, p2 = vertex2.position;
			float3 bary = sampleUniformTriangle(u2);

			float2 uv = (bary.x * vertex0.tex_coords) + (bary.y * vertex1.tex_coords) + (bary.z * vertex2.tex_coords);
			float3 p = (p0 * bary.x) + (p1 * bary.y) + (p2 * bary.z);
			p = make_float3(model * make_float4(p, 1));
			float3 geo_normal = make_float3(model * make_float4(local_geometric_normal, 0));

			float pdf = 1.0f / getArea();
			pdf *= Sqr(distance(p, ctx.wpos));
			pdf /= AbsDot(normalize(p - ctx.wpos), geo_normal);

			return ShapeSample(p, geo_normal, pdf, uv);
		};

		__device__ float getArea() const {
			return 0.5f * length(cross(vertex1.position - vertex0.position, vertex2.position - vertex0.position));
		};

	public:
		Vertex vertex0, vertex1, vertex2;
		float3 local_geometric_normal{ 0.0f,0.0f,0.0f };
		int32_t material_id = -1;
		int32_t light_id = -1;
		int32_t mesh_id = -1;
	};

	class TriangleMesh
	{
	public:
		TriangleMesh(int prim_offset, int prim_count, Mat4 inv_model) :
			prim_offset(prim_offset),
			prim_count(prim_count),
			curr_inv_model_matrix(inv_model),
			prev_inv_model_matrix(inv_model)
		{};

		void setTransform(const Mat4& inv_model)
		{
			prev_inv_model_matrix = curr_inv_model_matrix;
			curr_inv_model_matrix = inv_model;
		};

		Mat4 curr_inv_model_matrix;
		Mat4 prev_inv_model_matrix;

		int32_t blas_index = -1;
		uint32_t prim_count = 0;
		int32_t prim_offset = -1;
	};
}/*KittlesPT*/