#include "camera.cuh"
#include "ray.cuh"

namespace KittlesPT
{
	__device__ Ray Camera::generateRay(float2 ndc_coords, int2 frame_resolution) const
	{
		const float aspect_ratio = (float)frame_resolution.x / (float)frame_resolution.y;

		constexpr float camera_height = 2.0f;
		const float camera_width = aspect_ratio * camera_height;

		float4 target_cs = inv_projection_matrix * make_float4(ndc_coords.x, ndc_coords.y, 1.f, 1.f);
		float4 target_ws = inv_view_matrix * make_float4(normalize(make_float3(target_cs) / target_cs.w), 0);
		float3 raydir_ws = normalize(make_float3(target_ws));
		float3 rayorig_ws = make_float3(inv_view_matrix * make_float4(0, 0, 0, 1));

		//-1 => forward depth
		Ray ray = Ray(rayorig_ws, raydir_ws);
		return ray;
	}
	__host__ void Camera::setView(Mat4 inv_proj, Mat4 inv_view)
	{
		//Mat4::print_matrix(inv_view);
		inv_projection_matrix = inv_proj;
		inv_view_matrix = inv_view;
		world_position = make_float3(inv_view[3]);
	}

	//====================================================================================================================

	namespace AgxMinimal
	{
		// 0: Default, 1: Golden, 2: Punchy
#define AGX_LOOK 0

//Fifth order
// Mean error^2: 3.6705141e-06
		__device__ float3 agxDefaultContrastApprox(float3 x) {
			float3 x2 = x * x;
			float3 x4 = x2 * x2;

			return +15.5 * x4 * x2
				- 40.14 * x4 * x
				+ 31.96 * x4
				- 6.868 * x2 * x
				+ 0.4298 * x2
				+ 0.1191 * x
				- 0.00232;
		}

		__device__ float3 agx_fitted(float3 col) {
			float3 val = (col);
			const Mat3 agx_mat = Mat3(
				0.842479062253094, 0.0423282422610123, 0.0423756549057051,
				0.0784335999999992, 0.878468636469772, 0.0784336,
				0.0792237451477643, 0.0791661274605434, 0.879142973793104);

			const float min_ev = -12.47393f;
			const float max_ev = 4.026069f;

			// Input transform (inset)
			val = agx_mat * val;

			// Log2 space encoding
			val = clamp(log2f(val), min_ev, max_ev);
			val = (val - min_ev) / (max_ev - min_ev);

			// Apply sigmoid function approximation
			val = agxDefaultContrastApprox(val);

			return float3(val);
		}

		__device__ float3 agx_fitted_Eotf(float3 col) {
			float3 val = (col);
			const Mat3 agx_mat_inv = Mat3(
				1.19687900512017, -0.0528968517574562, -0.0529716355144438,
				-0.0980208811401368, 1.15190312990417, -0.0980434501171241,
				-0.0990297440797205, -0.0989611768448433, 1.15107367264116);

			// Inverse input transform (outset)
			val = agx_mat_inv * val;

			// sRGB IEC 61966-2-1 2.2 Exponent Reference EOTF Display
			// NOTE: We're linearizing the output here. Comment/adjust when
			// *not* using a sRGB render target
			val = powf(val, make_float3(2.2));

			return float3(val);
		}

		__device__ float3 agxLook(float3 val)
		{
			const float3 lw = make_float3(0.2126, 0.7152, 0.0722);
			float luma = dot(val, lw);

			// Default
			float3 offset = make_float3(0.0);
			float3 slope = make_float3(1.0);
			float3 power = make_float3(1.0);
			float sat = 1.0;

#if AGX_LOOK == 1
			// Golden
			slope = make_float3(1.0, 0.9, 0.5);
			power = make_float3(0.8);
			sat = 0.8;
#elif AGX_LOOK == 2
			// Punchy
			slope = make_float3(1.0);
			power = make_float3(1.35, 1.35, 1.35);
			sat = 1.4;
#endif

			// ASC CDL
			val = powf(val * slope + offset, power);
			return luma + sat * (val - luma);
		}
	}

	//============================================================================================

	__device__ float3 Film::getDisplayRGB(RGBSpectrum HDR_linear_radiance) const
	{
		float3 display_color = AgxMinimal::agx_fitted(HDR_linear_radiance.toFloat3());
		display_color = AgxMinimal::agxLook(display_color);
		display_color = AgxMinimal::agx_fitted_Eotf(display_color);

		return display_color;
	}
}