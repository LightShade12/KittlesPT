#include "camera.cuh"

namespace KittlesPT
{
	__host__ void Camera::setView(Mat4 inv_proj, Mat4 inv_view)
	{
		inv_projection_matrix = inv_proj;
		inv_view_matrix = inv_view;
		world_position = make_float3(inv_view[3]);
	}

	__host__ void Camera::setExposure(float luminance_exposure_scalar, float white_point_ev, float black_point_ev)
	{
		film.luminance_exposure_scalar = luminance_exposure_scalar;
		film.white_point_ev = white_point_ev;
		film.black_point_ev = black_point_ev;
	}

	//====================================================================================================================

	/*
	* AgX Minimal Implementation
	Courtesy: Benjamin Wrensch
	From: https://iolite-engine.com/blog_posts/minimal_agx_implementation
	*/
	namespace AgXMinimal
	{
		/*namespace BaseValues {
			__constant__ constexpr Mat3 g_AgX_mat = Mat3(
				0.842479062253094f, 0.0423282422610123f, 0.0423756549057051f,
				0.0784335999999992f, 0.878468636469772f, 0.0784336f,
				0.0792237451477643f, 0.0791661274605434f, 0.879142973793104f);

			__constant__ constexpr Mat3 g_AgX_mat_inv = Mat3(
				1.19687900512017f, -0.0528968517574562f, -0.0529716355144438f,
				-0.0980208811401368f, 1.15190312990417f, -0.0980434501171241f,
				-0.0990297440797205f, -0.0989611768448433f, 1.15107367264116f);

			const float min_ev = -12.47393f;
			const float max_ev = 4.026069f;
		}*/

		/*
		* Modified AgX (closer to Blender)
		* Courtesy: https://github.com/Calinou
		* From: https://github.com/godotengine/godot/blob/master/servers/rendering/renderer_rd/shaders/effects/tonemap.glsl
		*/

		__constant__ constexpr Mat3 LINEAR_SRGB_TO_LINEAR_REC2020 = Mat3(
			0.6274f, 0.0691f, 0.0164f,
			0.3293f, 0.9195f, 0.0880f,
			0.0433f, 0.0113f, 0.8956f);

		__constant__ constexpr Mat3 AgX_INSET_MATRIX = Mat3(
			0.856627153315983f, 0.137318972929847f, 0.11189821299995f,
			0.0951212405381588f, 0.761241990602591f, 0.0767994186031903f,
			0.0482516061458583f, 0.101439036467562f, 0.811302368396859f);

		__constant__ constexpr Mat3 AgX_OUTSET_REC2020_TO_sRGB_MATRIX = Mat3(
			1.9648846919172409596f, -0.29937618452442253746f, -0.16440106280678278299f,
			-0.85594737466675834968f, 1.3263980951083531115f, -0.23819967517076844919f,
			-0.10883731725048386702f, -0.02702191058393112346f, 1.4025007379775505276f);

		__constant__ constexpr float3 LUMINANCE_COEFFICIENTS{ 0.2126f, 0.7152f, 0.0722f };//spectral curve coefficients
		__constant__ constexpr float MIDDLE_GRAY = 0.18f;

		// 0: Default, 1: Golden, 2: Punchy
#define AGX_LOOK 0

		// ASC CDL based look transform
		__device__ float3 AgXLook(float3 val)
		{
			float luma = dot(val, LUMINANCE_COEFFICIENTS);

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

			// ASC CDL based look transform
			val = powf(val * slope + offset, power);
			return luma + sat * (val - luma);
		}

		//Fifth order
		//Mean error^2: 3.6705141e-06
		__device__ inline float3 AgXDefaultContrastApprox(float3 x)
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

		//Input is expected as linear tristimulus with Rec.709(BT 709) primary chromaticities ("linear sRGB")
		__device__ float3 AgXFitted(float3 linear_rec_709, float white_point_ev, float black_point_ev)
		{
			/*From https://gist.github.com/nxrighthere/eb208dae8b66dbe452af223f276e46cc
			// DEFAULT_LOG2_MIN      = -10.0
			// DEFAULT_LOG2_MAX      =  +6.5
			// MIDDLE_GRAY           =  0.18
			// log2(pow(2, VALUE) * MIDDLE_GRAY)
			// Adjusted for Unreal's zero exposure compensation
			const float min_ev = -12.47393f; // Default: -12.47393f;
			const float max_ev = 0.526069f;  // Default:  4.026069f;
			*/

			const float AgX_min_ev = log2(pow(2, black_point_ev) * MIDDLE_GRAY);
			const float AgX_max_ev = log2(pow(2, white_point_ev) * MIDDLE_GRAY);
			const float dynamic_range = AgX_max_ev - AgX_min_ev;

			//AgX in Rec 2020 to match Blender better
			linear_rec_709 = LINEAR_SRGB_TO_LINEAR_REC2020 * linear_rec_709;
			//prevent -ve values for AgX inset; loss of information if done before REC2020 transform
			linear_rec_709 = fmaxf(linear_rec_709, make_float3(0.0f));

			// Input transform (inset)
			linear_rec_709 = AgX_INSET_MATRIX * linear_rec_709;

			//linear_rec_709 = BaseValues::g_AgX_mat * linear_rec_709;

			// Log2 space encoding
			float3 log2_rec_709 = clamp(log2f(linear_rec_709), AgX_min_ev, AgX_max_ev);
			log2_rec_709 = (log2_rec_709 - AgX_min_ev) / dynamic_range;//normalization

			// Apply sigmoid function approximation
			float3 color_out = AgXDefaultContrastApprox(log2_rec_709);

			return color_out;
		}

		//Outputs NON-LINEAR Rec. 709
		__device__ float3 AgXFittedOETF(float3 val)
		{
			// Convert back to linear before applying outset matrix.
			val = powf(val, make_float3(2.4));

			// Inverse input transform (outset)
			//float3 non_linear_rec_709 = BaseValues::g_AgX_mat_inv * val;

			// Apply outset to make the result more chroma-laden and then go back to linear sRGB.
			float3 non_linear_rec_709 = AgX_OUTSET_REC2020_TO_sRGB_MATRIX * val;

			// sRGB IEC 61966-2-1 2.2 Exponent Reference EOTF Display
			// NOTE: We're linearizing the output here. Comment/adjust when
			// *not* using a sRGB render target
			//non_linear_rec_709 = powf(non_linear_rec_709, make_float3(2.2f));

			// sRGB approx OETF
			non_linear_rec_709 = fmaxf(non_linear_rec_709, make_float3(0.0f));
			non_linear_rec_709 = powf(non_linear_rec_709,
				constexpr_float3(0.45454545454545453f));//sRGB OETF approx (1.0/2.2)

			return non_linear_rec_709;
		}
	}

	//FILM============================================================================================

	__device__ float3 Film::getDisplayNonLinearSRGB(RGBSpectrum linear_radiance) const
	{
		float3 display_color = AgXMinimal::AgXFitted(linear_radiance.toFloat3(), white_point_ev, black_point_ev);
		display_color = AgXMinimal::AgXLook(display_color);
		display_color = AgXMinimal::AgXFittedOETF(display_color);
		//NOTE: display_color in NOT sRGB; its Rec. 709 with gamma 2.2(unlike usual 2.4); highly similar, different OETF however
		return display_color;
	}
}/*KittlesPT*/