#include "camera.cuh"
#include "Ray.cuh"

namespace KittlesPT
{
	__device__ Ray Camera::getRay(float2 ndc_coords, int2 frame_resolution) const
	{
		float aspect_ratio = (float)frame_resolution.x / (float)frame_resolution.y;

		float camera_height = 2.0f;
		float camera_width = aspect_ratio * camera_height;

		//-1 => forward depth
		float3 raydir = make_float3(ndc_coords.x * camera_width, ndc_coords.y * camera_height, -1);
		Ray ray = Ray(world_position, normalize(raydir));
		return ray;
	};
}