#pragma once
#include "shaders/triangle.cuh"
#include "shaders/blas.cuh"

#include <thrust/universal_vector.h>
#include <thrust/host_vector.h>
#include <numeric>

namespace KittlesPT
{
	class BLASBuilder
	{
	public:

		struct BVHTriangleCache {
			BVHTriangleCache(float3 centroid) :centroid(centroid) {};
			float3 centroid;
		};

		__host__ BLASBuilder(const thrust::universal_vector<Triangle>& tris, std::vector<int32_t>* trisid, std::vector<BVHNode>* bvhnodes) :
			m_bvhnodes_buffer(bvhnodes), m_tris_index_buffer(trisid)
		{
			m_tris_buffer = tris;
		}

		__host__ BLAS build(const TriangleMesh& mesh, int32_t mesh_id)
		{
			mesh_tris_count = mesh.prim_count;
			mesh_tris_offset = mesh.prim_offset;
			BLAS mesh_blas = buildBLAS();
			mesh_blas.inv_model_matrix = mesh.inv_model_matrix;
			mesh_blas.mesh_id = mesh_id;
			return mesh_blas;
		}
	private:

		__host__ void updateNodeBounds(uint32_t node_idx)
		{
			BVHNode& node = (*m_bvhnodes_buffer)[node_idx];
			node.bounds.pmin = make_float3(FLT_MAX);
			node.bounds.pmax = make_float3(-FLT_MAX);
			for (uint32_t first = node.left_child_node_id_or_tris_index_start_id, i = 0; i < node.node_tris_idx_count; i++)
			{
				uint32_t tri_id = (*m_tris_index_buffer)[first + i];
				Triangle& leaf_tri = m_tris_buffer[tri_id];
				node.bounds.grow(leaf_tri.vertex0.position);
				node.bounds.grow(leaf_tri.vertex1.position);
				node.bounds.grow(leaf_tri.vertex2.position);
			}
		}

		enum PlaneAxis {
			X,
			Y,
			Z
		};

		__host__ float comp(float3 v, int32_t idx)
		{
			return(idx == 0) ? v.x : (idx == 1) ? v.y : v.z;
		}

		__host__ float EvaluateSAH(const std::vector<BVHTriangleCache>& cache, BVHNode& node, PlaneAxis axis, float pos)
		{
			// determine triangle counts and bounds for this split candidate
			Bounds3f leftBox{}, rightBox{};
			int leftCount = 0, rightCount = 0;
			for (uint i = 0; i < node.node_tris_idx_count; i++)
			{
				int32_t tri_id = (*m_tris_index_buffer)[node.left_child_node_id_or_tris_index_start_id + i];
				const Triangle& triangle = m_tris_buffer[tri_id];
				const BVHTriangleCache& tri_cache = cache[tri_id];

				if (comp(tri_cache.centroid, axis) < pos)
				{
					leftCount++;
					leftBox.grow(triangle.vertex0.position);
					leftBox.grow(triangle.vertex1.position);
					leftBox.grow(triangle.vertex2.position);
				}
				else
				{
					rightCount++;
					rightBox.grow(triangle.vertex0.position);
					rightBox.grow(triangle.vertex1.position);
					rightBox.grow(triangle.vertex2.position);
				}
			}
			float cost = leftCount * leftBox.surfaceArea() + rightCount * rightBox.surfaceArea();
			return (cost > 0.0f) ? cost : INFINITY;
		}

		__host__ void subdivide(const std::vector<BVHTriangleCache>& cache, uint32_t node_id, uint32_t* node_index_ptr)
		{
			BVHNode& parent_node = (*m_bvhnodes_buffer)[node_id];

			int32_t best_axis = -1;
			float best_pos = 0, best_cost = INFINITY;
			for (int32_t axis = 0; axis < 3; axis++) {
				for (uint32_t i = 0; i < parent_node.node_tris_idx_count; i++)
				{
					int32_t tri_id = (*m_tris_index_buffer)[parent_node.left_child_node_id_or_tris_index_start_id + i];
					Triangle& triangle = m_tris_buffer[tri_id];
					float candidate_pos = comp(cache[tri_id].centroid, axis);
					float cost = EvaluateSAH(cache, parent_node, (PlaneAxis)axis, candidate_pos);
					if (cost < best_cost) {
						best_pos = candidate_pos, best_axis = axis, best_cost = cost;
					}
				}
			}
			int32_t axis = best_axis;
			float split_pos = best_pos;

			float parent_cost = parent_node.node_tris_idx_count * parent_node.surfaceArea();
			if (parent_cost <= best_cost) {
				return;//termination condition
			}

			//split grp---------
			int32_t i = parent_node.left_child_node_id_or_tris_index_start_id;
			int32_t j = i + parent_node.node_tris_idx_count - 1;
			while (i <= j)
			{
				if (comp(cache[(*m_tris_index_buffer)[i]].centroid, axis) < split_pos) {
					i++;
				}
				else {
					std::swap((*m_tris_index_buffer)[i], (*m_tris_index_buffer)[j--]);
				}
			}
			int32_t split_id = i;

			//create children------------
			BVHNode left, right;
			left.node_tris_idx_count = split_id - parent_node.left_child_node_id_or_tris_index_start_id;
			if (left.node_tris_idx_count == 0 || left.node_tris_idx_count == parent_node.node_tris_idx_count) {
				return;//full imbalanced split
			}
			int32_t left_child_id = (*node_index_ptr)++;
			int32_t right_child_id = (*node_index_ptr)++;
			left.left_child_node_id_or_tris_index_start_id = parent_node.left_child_node_id_or_tris_index_start_id;
			right.left_child_node_id_or_tris_index_start_id = split_id;
			right.node_tris_idx_count = parent_node.node_tris_idx_count - left.node_tris_idx_count;
			parent_node.node_tris_idx_count = 0;//mark parent as interior
			parent_node.left_child_node_id_or_tris_index_start_id = left_child_id;

			(*m_bvhnodes_buffer)[left_child_id] = left;
			(*m_bvhnodes_buffer)[right_child_id] = right;
			updateNodeBounds(left_child_id);
			updateNodeBounds(right_child_id);

			subdivide(cache, left_child_id, node_index_ptr);
			subdivide(cache, right_child_id, node_index_ptr);
		}

		__host__ BLAS buildBLAS()
		{
			std::vector<BVHTriangleCache>tris_cache;
			for (int32_t i = 0; i < m_tris_buffer.size(); i++) {
				Triangle& tri = m_tris_buffer[i];
				float3 centroid = (tri.vertex0.position + tri.vertex1.position + tri.vertex2.position) * 0.3333f;
				tris_cache.push_back(BVHTriangleCache(centroid));
			}

			m_bvhnodes_buffer->resize(2 * m_tris_buffer.size() - 1);
			m_tris_index_buffer->resize(m_tris_buffer.size());
			std::iota(m_tris_index_buffer->begin(), m_tris_index_buffer->end(), 0);

			BLAS blas;
			blas.bvhnode_root_id = 0;
			BVHNode& root = (*m_bvhnodes_buffer)[blas.bvhnode_root_id];
			root.node_tris_idx_count = m_tris_buffer.size();
			root.left_child_node_id_or_tris_index_start_id = 0;
			updateNodeBounds(blas.bvhnode_root_id);
			uint32_t node_index_ptr = 1;
			subdivide(tris_cache, blas.bvhnode_root_id, &node_index_ptr);
			m_bvhnodes_buffer->shrink_to_fit();
			return blas;
		}

		int32_t mesh_tris_offset = -1;
		uint32_t mesh_tris_count = 0;

		thrust::host_vector<Triangle> m_tris_buffer;
		std::vector<int32_t>* m_tris_index_buffer = nullptr;
		std::vector<BVHNode>* m_bvhnodes_buffer = nullptr;
	};

	class TLASBuilder
	{
	public:

		TLASBuilder(const thrust::universal_vector<BLAS>* blas_buffer) :
			blas_buffer(blas_buffer)
		{}
		const thrust::universal_vector<BLAS>* blas_buffer;

		__host__ int32_t findBestMatch(int32_t* TLASwork_idx_list, int32_t work_idx_list_size, int32_t TLAS_A_idx, const std::vector<TLASNode>& tlasnodes)
		{
			float smallest = FLT_MAX;
			int bestB = -1;
			for (int B = 0; B < work_idx_list_size; B++) {
				if (B != TLAS_A_idx)
				{
					float3 bmax = fmaxf(tlasnodes[TLASwork_idx_list[TLAS_A_idx]].bounds.pmax, tlasnodes[TLASwork_idx_list[B]].bounds.pmax);
					float3 bmin = fminf(tlasnodes[TLASwork_idx_list[TLAS_A_idx]].bounds.pmin, tlasnodes[TLASwork_idx_list[B]].bounds.pmin);
					float3 e = bmax - bmin;
					float surfaceArea = e.x * e.y + e.y * e.z + e.z * e.x;//half SA
					if (surfaceArea < smallest) {
						smallest = surfaceArea, bestB = B;
					}
				}
			}
			return bestB;
		}

		__host__ TLAS build(std::vector<TLASNode>* tlasnodes_buffer)
		{
			uint32_t BLAS_COUNT = blas_buffer->size();
			if (BLAS_COUNT <= 0) {
				return TLAS();
			}

			int32_t* tlas_node_ids = new int32_t[BLAS_COUNT];

			int32_t node_indices = BLAS_COUNT;//work list size

			// assign a TLASleaf parent_node to each BLAS; making work list
			for (uint64_t i = 0; i < BLAS_COUNT; i++)
			{
				tlas_node_ids[i] = BLAS_COUNT;//i derived from m_BLASCount also works
				TLASNode tlas_node;
				tlas_node.bounds = (*blas_buffer)[i].bounds;
				tlas_node.blas_id = i;
				tlas_node.left_right_id = 0u; // makes it a leaf
				tlasnodes_buffer->push_back(tlas_node);
			}

			// use agglomerative clustering to build the TLAS
			int A = 0, B = findBestMatch(tlas_node_ids, node_indices, A, *tlasnodes_buffer);
			while (node_indices > 1)
			{
				int C = findBestMatch(tlas_node_ids, node_indices, B, *tlasnodes_buffer);
				if (A == C)
				{
					//left | right
					int node_id_A = tlas_node_ids[A], node_id_B = tlas_node_ids[B];
					const TLASNode* nodeA = &((*tlasnodes_buffer)[node_id_A]);
					const TLASNode* nodeB = &((*tlasnodes_buffer)[node_id_B]);
					TLASNode new_node;
					new_node.left_right_id = node_id_A + (node_id_B << 16);
					new_node.bounds.pmin = fminf(nodeA->bounds.pmin, nodeB->bounds.pmin);
					new_node.bounds.pmax = fmaxf(nodeA->bounds.pmax, nodeB->bounds.pmax);
					tlasnodes_buffer->push_back(new_node);
					tlas_node_ids[A] = tlasnodes_buffer->size() - 1;
					tlas_node_ids[B] = tlas_node_ids[node_indices - 1];
					B = findBestMatch(tlas_node_ids, --node_indices, A, *tlasnodes_buffer);
				}
				else {
					A = B, B = C;
				}
			}
			//TODO:make and add root somewhere
			//tlasnodes[0] = tlasnodes[tlas_node_ids[A]];
			tlasnodes_buffer->push_back((*tlasnodes_buffer)[tlas_node_ids[A]]);

			delete[] tlas_node_ids;

			return TLAS();
		}
	};
}