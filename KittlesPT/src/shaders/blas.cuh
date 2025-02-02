#pragma once
#include "maths/bounds.cuh"
#include "ray.cuh"
#include "triangle.cuh"
#include "containers.cuh"

namespace KittlesPT
{
	class BVHNode
	{
	public:
		BVHNode() = default;

		__host__ float surfaceArea() const
		{
			if (node_tris_idx_count == 0) {
				return 0.0f;
			}
			return bounds.surfaceArea();
		}

		__host__ __device__ bool isLeaf() const
		{
			return node_tris_idx_count > 0;
		}

		__device__ bool intersectP(const Ray& ray, float tmin, float tmax, float* hit0, float* hit1)  const
		{
			return bounds.intersectP(ray, tmin, tmax, hit0, hit1);
		}

		int32_t left_child_node_id_or_tris_index_start_id = -1;
		int32_t node_tris_idx_count = 0;
		Bounds3f bounds;
	};

	__constant__ constexpr uint8_t BLAS_TRAVERSAL_MAX_STACK_DEPTH = 16;

	//One BLAS = One BVH Tree
	class BLAS
	{
	public:
		BLAS() = default;

		bool intersectP()
		{
			return false;
		};

		Intersection intersect(const GlobalShaderData& shader_data, const Ray& ray, float tmin, float tmax) const
		{
			if (bvhnode_root_id < 0) {
				return Intersection();
			}
			const BVHNode* bvh_nodes_buffer = nullptr;//put actual buffer here
			const int32_t* bvh_tri_ids_buffer = nullptr;//put actual buffer here

			Ray object_ray = ray.transform(inv_model_matrix);

			uint32_t node_id_stack[BLAS_TRAVERSAL_MAX_STACK_DEPTH]{};
			uint8_t stack_ptr = 0u;//max val=255
			node_id_stack[stack_ptr++] = bvhnode_root_id;

			const BVHNode* stack_top_node = &bvh_nodes_buffer[bvhnode_root_id];//load root node;unneded

			Intersection intr;
			Intersection closest;

			float child1_hitdist = INFINITY;
			float child2_hitdist = INFINITY;
			const Triangle* primitive = nullptr;

			//traversal
			while (stack_ptr > 0)
			{
				stack_top_node = &bvh_nodes_buffer[node_id_stack[--stack_ptr]];//pop node id stack

				//if interior
				if (!stack_top_node->isLeaf())
				{
					float c1exit = 0, c2exit = 0;
					child1_hitdist = INFINITY, child2_hitdist = INFINITY;
					bool c1hit = bvh_nodes_buffer[stack_top_node->left_child_node_id_or_tris_index_start_id].intersectP(object_ray,
						tmin, tmax, &child1_hitdist, &c1exit);
					bool c2hit = bvh_nodes_buffer[stack_top_node->left_child_node_id_or_tris_index_start_id + 1].intersectP(object_ray,
						tmin, tmax, &child2_hitdist, &c2exit);

					if (child1_hitdist > child2_hitdist) {
						if (c1hit && child1_hitdist < closest.distance) {
							node_id_stack[stack_ptr++] = stack_top_node->left_child_node_id_or_tris_index_start_id;
						}
						if (c2hit && child2_hitdist < closest.distance) {
							node_id_stack[stack_ptr++] = stack_top_node->left_child_node_id_or_tris_index_start_id + 1;
						}
					}
					else {
						if (c2hit && child2_hitdist < closest.distance) {
							node_id_stack[stack_ptr++] = stack_top_node->left_child_node_id_or_tris_index_start_id + 1;
						}
						if (c1hit && child1_hitdist < closest.distance) {
							node_id_stack[stack_ptr++] = stack_top_node->left_child_node_id_or_tris_index_start_id;
						}
					}
				}
				else
				{
					for (int32_t prim_index_id = stack_top_node->left_child_node_id_or_tris_index_start_id;
						prim_index_id < stack_top_node->left_child_node_id_or_tris_index_start_id + stack_top_node->node_tris_idx_count;
						prim_index_id++)
					{
						int32_t prim_id = bvh_tri_ids_buffer[prim_index_id];
						primitive = &shader_data.triangles_buffer.data[prim_id];
						primitive->intersect(object_ray, tmin, tmax, &intr);

						if (intr.distance < INFINITY && intr.distance < tmax && intr.distance > tmin && intr.distance < closest.distance)
						{
							closest = intr;
							closest.primitive_id = prim_id;
							closest.instance_id = mesh_id;
							//tmax = closest.distance;
						}
					}
				}
			}

			return closest;
		};

	public:
		Bounds3f bounds;
		int64_t bvhnode_root_id = -1;
		uint64_t mesh_id = -1;
		Mat4 inv_model_matrix;
	};

	class TLASNode {
	public:

		TLASNode() = default;

		Bounds3f bounds;
		uint32_t left_right_id = 0;
		int32_t blas_id = -1;

		__device__ __host__ bool isleaf() const {
			return left_right_id == 0;
		};

		//each child can represent max id of 65535
		__device__ void getChildrenID(uint32_t* child1_id, uint32_t* child2_id) const
		{
			*child1_id = left_right_id & 0xFFFF;//left child id in lower 16 bits
			*child2_id = (left_right_id >> 16) & 0xFFFF;//right child in higher 16 bits
		}

		__host__ float surfaceArea() const
		{
			return bounds.surfaceArea();
		}

		__device__ bool intersectP(const Ray& ray, float tmin, float tmax, float* hit0, float* hit1) const
		{
			return bounds.intersectP(ray, tmin, tmax, hit0, hit1);
		}
	};

	__constant__ constexpr uint8_t TLAS_TRAVERSAL_MAX_STACK_DEPTH = 16;//can handle max 65535 blases
	class TLAS {
	public:
		TLAS() = default;

		__device__ bool intersectP(const Ray& ray, float tmin, float tmax) const
		{
			return false;
		};

		__device__ Intersection intersect(const GlobalShaderData& shader_data, const Ray& ray, float tmin, float tmax) const
		{
			if (tlasnode_root_id < 0) {
				return Intersection();
			}
			const TLASNode* tlas_nodes_buffer = nullptr;//put actual buffer here
			const BLAS* blas_buffer = nullptr;

			int32_t node_id_stack[TLAS_TRAVERSAL_MAX_STACK_DEPTH]{};
			uint16_t stack_ptr = 0u;

			const TLASNode* stack_top_node = &tlas_nodes_buffer[tlasnode_root_id];//load root; unneeded
			node_id_stack[stack_ptr++] = tlasnode_root_id;

			float child1_hitdist = INFINITY;
			float child2_hitdist = INFINITY;
			Intersection closest;

			//traversal
			while (stack_ptr > 0) {
				stack_top_node = &tlas_nodes_buffer[node_id_stack[--stack_ptr]];

				//if interior
				if (!stack_top_node->isleaf())
				{
					uint32_t c1id = 0u, c2id = 0u;
					stack_top_node->getChildrenID(&c1id, &c2id);

					float c1exit = 0, c2exit = 0;
					bool c1hit = tlas_nodes_buffer[c1id].intersectP(ray, tmin, tmax, &child1_hitdist, &c1exit);
					bool c2hit = tlas_nodes_buffer[c2id].intersectP(ray, tmin, tmax, &child2_hitdist, &c2exit);

					//TODO:implement early cull properly see discord for ref
					if (child1_hitdist > child2_hitdist) {
						if (c1hit && child1_hitdist < closest.distance) {
							node_id_stack[stack_ptr++] = c1id;
						}
						if (c2hit && child2_hitdist < closest.distance) {
							node_id_stack[stack_ptr++] = c2id;
						}
					}
					else {
						if (c2hit && child2_hitdist < closest.distance) {
							node_id_stack[stack_ptr++] = c2id;
						}
						if (c1hit && child1_hitdist < closest.distance) {
							node_id_stack[stack_ptr++] = c1id;
						}
					}
				}
				else//if leaf
				{
					closest = blas_buffer[stack_top_node->blas_id].intersect(shader_data, ray, tmin, tmax);
				}
			}
			return closest;
		}

	public:
		Bounds3f bounds;
		int32_t tlasnode_root_id = -1;
	};
}