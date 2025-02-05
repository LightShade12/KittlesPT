#pragma once
#include "maths/linear_algebra.cuh"
#include "color.cuh"

namespace KittlesPT
{
	class Ray;

	//Not needed but added for parity with PBRTv4 implementation
	class Film
	{
	public:
		__device__ float3 getDisplayNonLinearSRGB(RGBSpectrum linear_radiance) const;

		float luminance_exposure_scalar = 1.0f;//TODO: fix this retarded shit
		float black_point_ev = -10.0f;
		float white_point_ev = 6.5f;
	};

	class Camera
	{
	public:

		Camera() = default;

		//generate camera rays; -1 => forawrd depth
		__device__ Ray generateRay(float2 ndc_coords) const;

		__host__ void setView(Mat4 inv_proj, Mat4 inv_view);
		__host__ void setExposure(float luminance_exposure_scalar, float white_point_ev, float black_point_ev);

	public:
		Film film;
		Mat4 inv_view_matrix;
		Mat4 inv_projection_matrix;
		float3 world_position;
	};
}/*KittlesPT*/