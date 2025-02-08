#pragma once
#include "maths/linear_algebra.cuh"
#include "ray.cuh"
#include "color.cuh"

namespace KittlesPT
{
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
		__device__ Ray generateRay(float2 ndc_coords) const
		{
			float4 target_cs = inv_projection_matrix * make_float4(ndc_coords.x, ndc_coords.y, 1.0f, 1.0f);
			float4 target_ws = inv_view_matrix * make_float4(normalize(make_float3(target_cs) / target_cs.w), 0.0f);
			float3 raydir_ws = normalize(make_float3(target_ws));
			float3 rayorig_ws = world_position;

			//Z = -1 => forward depth
			return Ray(rayorig_ws, raydir_ws);
		}

		__host__ void setView(Mat4 inv_proj, Mat4 inv_view);
		__host__ void setExposure(float luminance_exposure_scalar, float white_point_ev, float black_point_ev);

	public:
		Film film;
		Mat4 inv_view_matrix;
		Mat4 inv_projection_matrix;
		float3 world_position;
	};
}/*KittlesPT*/