#include "tlas.cuh"
#include "containers.cuh"
#include "blas.cuh"
#include "ray.cuh"

namespace KittlesPT
{
	__device__ bool TLAS::intersectP(const ShaderData& shader_data, const Ray& ray, float tmin, float tmax) const
	{
		if (tlasnode_root_index < 0) {
			return false;
		}
		float child1_hitdist = INFINITY, child2_hitdist = INFINITY;
		if (!bounds.intersectP(ray, tmin, tmax, &child1_hitdist, &child2_hitdist)) {
			return false;
		}
		const TLASNode* const tlas_nodes_buffer = shader_data.tlas_nodes_buffer.data;
		const BLAS2* const blas_buffer = shader_data.blas_buffer.data;

		int32_t node_indice_stack[TLAS_TRAVERSAL_MAX_STACK_DEPTH]{};
		uint16_t stack_ptr = 0u;

		const TLASNode* stack_top_node = &tlas_nodes_buffer[tlasnode_root_index];//load root; unneeded
		node_indice_stack[stack_ptr++] = tlasnode_root_index;

		bool hit = false;

		//traversal
		while (stack_ptr > 0) {
			stack_top_node = &tlas_nodes_buffer[node_indice_stack[--stack_ptr]];

			//if interior
			if (!stack_top_node->isleaf())
			{
				uint32_t c1_index = 0u, c2_index = 0u;
				stack_top_node->getChildrenIndex(&c1_index, &c2_index);

				float c1exit = 0, c2exit = 0;
				bool c1hit = tlas_nodes_buffer[c1_index].intersectP(ray, tmin, tmax, &child1_hitdist, &c1exit);
				bool c2hit = tlas_nodes_buffer[c2_index].intersectP(ray, tmin, tmax, &child2_hitdist, &c2exit);

				if (c1hit) {
					node_indice_stack[stack_ptr++] = c1_index;
				}
				if (c2hit) {
					node_indice_stack[stack_ptr++] = c2_index;
				}
			}
			else//if leaf
			{
				hit = blas_buffer[stack_top_node->blas_index].intersectP(shader_data, ray, tmin, tmax);
				if (hit) {
					return true;
				}
			}
		}
		return false;
	};

	__device__ Intersection TLAS::intersect(const ShaderData& shader_data, const Ray& ray, float tmin, float tmax, DebugData& dbg) const
	{
		if (tlasnode_root_index < 0) {
			return Intersection();
		}
		float child1_hitdist = INFINITY;
		float child2_hitdist = INFINITY;
		if (!bounds.intersectP(ray, tmin, tmax, &child1_hitdist, &child2_hitdist)) {
			return Intersection();
		}
		child1_hitdist = INFINITY;
		child2_hitdist = INFINITY;

		const TLASNode* const tlas_nodes_buffer = shader_data.tlas_nodes_buffer.data;
		const BLAS2* const blas_buffer = shader_data.blas_buffer.data;

		int32_t node_indice_stack[TLAS_TRAVERSAL_MAX_STACK_DEPTH]{};
		uint16_t stack_ptr = 0u;

		const TLASNode* stack_top_node = &tlas_nodes_buffer[tlasnode_root_index];//load root; unneeded
		node_indice_stack[stack_ptr++] = tlasnode_root_index;

		Intersection closest;

		//traversal
		while (stack_ptr > 0) {
			stack_top_node = &tlas_nodes_buffer[node_indice_stack[--stack_ptr]];
			dbg.tlas_hits++;
			//if interior
			if (!stack_top_node->isleaf())
			{
				uint32_t c1_index = 0u, c2_index = 0u;
				stack_top_node->getChildrenIndex(&c1_index, &c2_index);

				float c1exit = 0, c2exit = 0;
				bool c1hit = tlas_nodes_buffer[c1_index].intersectP(ray, tmin, tmax, &child1_hitdist, &c1exit);
				bool c2hit = tlas_nodes_buffer[c2_index].intersectP(ray, tmin, tmax, &child2_hitdist, &c2exit);

				//TODO:implement early cull properly see discord for ref
				if (child1_hitdist > child2_hitdist) {
					if (c1hit && child1_hitdist < closest.distance) {
						node_indice_stack[stack_ptr++] = c1_index;
					}
					if (c2hit && child2_hitdist < closest.distance) {
						node_indice_stack[stack_ptr++] = c2_index;
					}
				}
				else {
					if (c2hit && child2_hitdist < closest.distance) {
						node_indice_stack[stack_ptr++] = c2_index;
					}
					if (c1hit && child1_hitdist < closest.distance) {
						node_indice_stack[stack_ptr++] = c1_index;
					}
				}
			}
			else//if leaf
			{
				Intersection intr = blas_buffer[stack_top_node->blas_index].intersect(shader_data, ray, tmin, tmax, dbg);
				if (intr.distance < closest.distance) {
					closest = intr;
				}
			}
		}
		return closest;
	}
}