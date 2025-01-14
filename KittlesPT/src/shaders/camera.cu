#include "camera.cuh"
#include "ray.cuh"

namespace KittlesPT
{
	__device__ Ray Camera::generateRay(float2 ndc_coords, int2 frame_resolution) const
	{
		float4 target_cs = inv_projection_matrix * make_float4(ndc_coords.x, ndc_coords.y, 1.0f, 1.0f);
		float4 target_ws = inv_view_matrix * make_float4(normalize(make_float3(target_cs) / target_cs.w), 0.0f);
		float3 raydir_ws = normalize(make_float3(target_ws));
		float3 rayorig_ws = make_float3(inv_view_matrix * make_float4(0.0f, 0.0f, 0.0f, 1.0f));

		//-1 => forward depth
		return Ray(rayorig_ws, raydir_ws);
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
		__constant__ constexpr Mat3 g_AgX_mat = Mat3(
			0.842479062253094f, 0.0423282422610123f, 0.0423756549057051f,
			0.0784335999999992f, 0.878468636469772f, 0.0784336f,
			0.0792237451477643f, 0.0791661274605434f, 0.879142973793104f);

		__constant__ constexpr Mat3 g_AgX_mat_inv = Mat3(
			1.19687900512017f, -0.0528968517574562f, -0.0529716355144438f,
			-0.0980208811401368f, 1.15190312990417f, -0.0980434501171241f,
			-0.0990297440797205f, -0.0989611768448433f, 1.15107367264116f);

		__constant__ constexpr float3 g_luminance_weights = constexpr_float3(0.2126f, 0.7152f, 0.0722f);
		__constant__ constexpr float min_ev = -12.47393f;
		__constant__ constexpr float max_ev = 4.026069f;

		// 0: Default, 1: Golden, 2: Punchy
#define AGX_LOOK 0

		__device__ float3 agxLook(float3 val)
		{
			float luma = dot(val, g_luminance_weights);

			// Default
			float3 offset{ 0.0f,0.0f,0.0f };
			float3 slope{ 1.0f,1.0f,1.0f };
			float3 power{ 1.0f,1.0f,1.0f };
			float sat = 1.0f;

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

		//Fifth order
// Mean error^2: 3.6705141e-06
		__device__ float3 agxDefaultContrastApprox(float3 x)
		{
			float3 x2 = x * x;
			float3 x4 = x2 * x2;

			return +15.5f * x4 * x2
				- 40.14f * x4 * x
				+ 31.96f * x4
				- 6.868f * x2 * x
				+ 0.4298f * x2
				+ 0.1191f * x
				- 0.00232f;
		}

		__device__ float3 agx_fitted(float3 val)
		{
			// Input transform (inset)
			val = g_AgX_mat * val;

			// Log2 space encoding
			val = clamp(log2f(val), min_ev, max_ev);
			val = (val - min_ev) / (max_ev - min_ev);

			// Apply sigmoid function approximation
			val = agxDefaultContrastApprox(val);

			return val;
		}

		__device__ float3 agx_fitted_Eotf(float3 val)
		{
			// Inverse input transform (outset)
			val = g_AgX_mat_inv * val;

			// sRGB IEC 61966-2-1 2.2 Exponent Reference EOTF Display
			// NOTE: We're linearizing the output here. Comment/adjust when
			// *not* using a sRGB render target
			val = powf(val, make_float3(2.2f));

			return float3(val);
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
}/*KittlesPT*/