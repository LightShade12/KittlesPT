#pragma once
#include "../maths/linear_algebra.cuh"

namespace KittlesPT
{
	class Ray;

	class Film
	{
	public:
		__device__ float3 getDisplayL(float3 HDR_radiance) const;
	};

	class Camera
	{
	public:

		__host__ __device__ Camera() = default;
		__host__ __device__ Camera(float3 pos, float3 forward) :
			world_position(pos), forward_direction(forward) {};

		//generate camera rays; -1 => forawrd depth
		__device__ Ray generateRay(float2 ndc_coords, int2 frame_resolution) const;

		__host__ void setView(Mat4 inv_proj, Mat4 inv_view);

	public:
		Film film;
		Mat4 inv_view_matrix;
		Mat4 inv_projection_matrix;
		float3 world_position;
		float3 forward_direction;
	};
}