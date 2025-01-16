#pragma once
#include "maths/linear_algebra.cuh"
#include "color.cuh"

namespace KittlesPT
{
	class Ray;

	class Film
	{
	public:
		__device__ float3 getDisplayRGB(RGBSpectrum HDR_linear_radiance) const;

		float exposure_EV = 0.001f;
	};

	class Camera
	{
	public:

		Camera() = default;

		__host__ Camera(float3 pos) :
			world_position(pos) {};

		//generate camera rays; -1 => forawrd depth
		__device__ Ray generateRay(float2 ndc_coords, int2 frame_resolution) const;

		__host__ void setView(Mat4 inv_proj, Mat4 inv_view);

	public:
		Film film;
		Mat4 inv_view_matrix;
		Mat4 inv_projection_matrix;
		float3 world_position;
	};
}/*KittlesPT*/