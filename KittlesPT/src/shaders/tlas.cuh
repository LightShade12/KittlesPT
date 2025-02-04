#pragma once
#include "maths/bounds.cuh"
#include <cuda_runtime.h>

namespace KittlesPT
{
	struct GlobalShaderData;
	struct Intersection;

	__constant__ constexpr uint8_t TLAS_TRAVERSAL_MAX_STACK_DEPTH = 16;//can handle max 65535 blases
	class TLAS
	{
	public:
		TLAS() = default;

		__device__ bool intersectP(const GlobalShaderData& shader_data, const Ray& ray, float tmin, float tmax) const;

		__device__ Intersection intersect(const GlobalShaderData& shader_data, const Ray& ray, float tmin, float tmax) const;

	public:
		Bounds3f bounds;
		int32_t tlasnode_root_id = -1;
	};
}