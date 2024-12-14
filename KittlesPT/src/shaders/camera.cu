#include "camera.cuh"
#include "Ray.cuh"

namespace KittlesPT
{
	__device__ Ray Camera::generateRay(float2 ndc_coords, int2 frame_resolution) const
	{
		float aspect_ratio = (float)frame_resolution.x / (float)frame_resolution.y;

		float camera_height = 2.0f;
		float camera_width = aspect_ratio * camera_height;

		float4 target_cs = inv_projection_matrix * make_float4(ndc_coords.x, ndc_coords.y, 1.f, 1.f);
		float4 target_ws = inv_view_matrix * make_float4(
			normalize(make_float3(target_cs) / target_cs.w), 0);
		float3 raydir_ws = make_float3(target_ws);
		float3 rayorig_ws = make_float3(inv_view_matrix * make_float4(0, 0, 0, 1));

		//-1 => forward depth
		float3 raydir = make_float3(ndc_coords.x * camera_width, ndc_coords.y * camera_height, -1);
		//Ray ray = Ray(world_position, normalize(raydir));

		Ray ray = Ray(rayorig_ws, normalize(raydir_ws));
		return ray;
	}
	__host__ void Camera::setView(Mat4 inv_proj, Mat4 inv_view)
	{
		//Mat4::print_matrix(inv_view);
		inv_projection_matrix = inv_proj;
		inv_view_matrix = inv_view;
		world_position = make_float3(inv_view[3]);
	}
}