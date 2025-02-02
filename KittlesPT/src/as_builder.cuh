#pragma once
#include "shaders/triangle.cuh"
#include "shaders/blas.cuh"
#include <thrust/universal_vector.h>

namespace KittlesPT
{
	class BLASBuilder
	{
	public:
		__host__ BLASBuilder()
		{
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

		__host__ BLAS buildBLAS()
		{
		}

		int32_t mesh_tris_offset = -1;
		uint32_t mesh_tris_count = 0;

		thrust::universal_vector<Triangle> tris_buffer;
		thrust::universal_vector<int32_t> tris_index_buffer;
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
				return;
			}

			int32_t* tlas_node_ids = new int32_t[BLAS_COUNT];

			int32_t node_indices = BLAS_COUNT;//work list size

			// assign a TLASleaf node to each BLAS; making work list
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
		}
	};
}