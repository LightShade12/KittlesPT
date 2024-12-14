#pragma once
#include <vector_types.h>

namespace KittlesPT
{
	class Ray;

	class Camera
	{
	public:

		__host__ __device__ Camera() = default;
		__host__ __device__ Camera(float3 pos, float3 forward) :
			world_position(pos), forward_direction(forward) {};

		//generate camera rays; -1 => forawrd depth
		__device__ Ray getRay(float2 ndc_coords, int2 frame_resolution) const;

		float3 world_position;
		float3 forward_direction;
	};
}