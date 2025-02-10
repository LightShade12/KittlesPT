#pragma once
#include "maths/linear_algebra.cuh"
#include "ray.cuh"
#include "film.cuh"
#include "color.cuh"

namespace KittlesPT
{
	class Camera
	{
	public:

		Camera() = default;

		//generate camera rays; -1 => forawrd depth
		__device__ Ray generateRay(float2 ndc_coords) const
		{
			float4 target_cs = curr_inv_projection_matrix * make_float4(ndc_coords.x, ndc_coords.y, 1.0f, 1.0f);
			float4 target_ws = curr_inv_view_matrix * make_float4(normalize(make_float3(target_cs) / target_cs.w), 0.0f);
			float3 raydir_ws = normalize(make_float3(target_ws));
			float3 rayorig_ws = curr_world_position;

			//Z = -1 => forward depth
			return Ray(rayorig_ws, raydir_ws);
		}
		__device__ Film getFilm() const { return m_film; }

		__host__ void Camera::setView(Mat4 inv_proj, Mat4 inv_view)
		{
			prev_inv_view_matrix = curr_inv_view_matrix;
			prev_inv_projection_matrix = curr_inv_projection_matrix;

			curr_inv_projection_matrix = inv_proj;
			curr_inv_view_matrix = inv_view;
			curr_world_position = make_float3(inv_view[3]);
		}

		__host__ void Camera::setExposure(float luminance_exposure_scalar, float white_point_ev, float black_point_ev)
		{
			m_film.luminance_exposure_scalar = luminance_exposure_scalar;
			m_film.white_point_ev = white_point_ev;
			m_film.black_point_ev = black_point_ev;
		}

	public:
		Mat4 curr_inv_view_matrix;
		Mat4 curr_inv_projection_matrix;
		Mat4 prev_inv_view_matrix;
		Mat4 prev_inv_projection_matrix;

		float3 curr_world_position;//TODO: redundant data
	private:
		Film m_film;
	};
}/*KittlesPT*/