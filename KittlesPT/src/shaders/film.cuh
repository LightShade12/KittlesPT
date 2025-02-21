#pragma once
#include "color.cuh"
#include "maths/linear_algebra.cuh"

namespace KittlesPT
{
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

		__constant__ constexpr Mat3 LINEAR_SRGB_TO_LINEAR_REC2020_MATRIX = Mat3(
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

		__constant__ constexpr float MIDDLE_GRAY = 0.18f;

		/// <summary>
		/// Fifth order
		/// <para/> Mean error^2: 3.6705141e-06
		/// </summary>
		/// <param name="x"> : normalized input values</param>
		/// <returns> normalized curve output </returns>
		inline __device__ float3 AgXDefaultContrastApprox(const float3& x)
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

		/// <summary>
		/// Input is expected as linear tristimulus with Rec.709(BT 709) primary chromaticities (exactly same as "linear sRGB" primaries)
		/// </summary>
		/// <param name="linear_rec_709"></param>
		/// <param name="white_point_ev"></param>
		/// <param name="black_point_ev"></param>
		/// <returns>Values adjusted with the sigmoid curve</returns>
		inline __device__ float3 AgXFitted(float3 linear_rec_709, float white_point_ev, float black_point_ev)
		{
			/*NOTES:
			From https://gist.github.com/nxrighthere/eb208dae8b66dbe452af223f276e46cc
			// DEFAULT_LOG2_MIN      = -10.0
			// DEFAULT_LOG2_MAX      =  +6.5
			// MIDDLE_GRAY           =  0.18
			// log2(pow(2, VALUE) * MIDDLE_GRAY)
			// Adjusted for Unreal's zero exposure compensation
			const float min_ev = -12.47393f; // Default: -12.47393f;
			const float max_ev = 0.526069f;  // Default:  4.026069f;
			*/

			const float AgX_min_ev = log2f(exp2f(black_point_ev) * MIDDLE_GRAY);
			const float AgX_max_ev = log2f(exp2f(white_point_ev) * MIDDLE_GRAY);
			const float dynamic_range = AgX_max_ev - AgX_min_ev;

			//AgX in Rec 2020 to match Blender better
			float3 linear_rec_2020 = LINEAR_SRGB_TO_LINEAR_REC2020_MATRIX * linear_rec_709;
			//prevent -ve values for AgX inset; loss of information if done before REC 2020 transform
			linear_rec_2020 = fmaxf(linear_rec_2020, make_float3(0.0f));

			// Input transform (inset)
			float3 agx_inset_linear_rec_2020 = AgX_INSET_MATRIX * linear_rec_2020;

			//linear_rec_709 = BaseValues::g_AgX_mat * linear_rec_709;

			// Log2 space encoding(in AgX inset space)
			float3 log2_rec_2020 = clamp(log2f(agx_inset_linear_rec_2020), AgX_min_ev, AgX_max_ev);
			float3 log2_rec_2020_normalized = (log2_rec_2020 - AgX_min_ev) / dynamic_range;//normalization

			// Apply sigmoid function approximation (tonemapping curve)
			float3 tonemapped_color = AgXDefaultContrastApprox(log2_rec_2020_normalized);

			return tonemapped_color;
		}

		// 0: Default, 1: Golden, 2: Punchy
#define AGX_LOOK 2

		/// <summary>
		/// ASC CDL based look transform
		/// </summary>
		/// <param name="val"> : normalized values from sigmoid curve</param>
		/// <returns>normalized look adjusted curve</returns>
		inline __device__ float3 AgXLook(const float3& val)
		{
			float luma = dot(val, rec709_luminance_coeffs);

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
			power = make_float3(1.05);//1.35
			sat = 1.25;//1.4
#endif

			// ASC CDL based look transform
			float3 new_val = powf(val * slope + offset, power);
			return luma + sat * (new_val - luma);
		}

		/// <summary>
		/// Outputs non-linear sRGB
		/// </summary>
		/// <param name="non_linear_rec2020"></param>
		/// <returns>Normalized non-linear sRGB color</returns>
		inline __device__ float3 AgXFittedEOTF(const float3& non_linear_rec2020)//how is it now in non-linear space??
		{
			// Convert back to linear before applying outset matrix.
			float3 linear_rec2020 = powf(non_linear_rec2020, make_float3(2.4));//approximate Rec 2020 decoding(gamma 2.4)

			// Inverse input transform (outset)
			//float3 linear_sRGB = BaseValues::g_AgX_mat_inv * val;

			// Apply outset to make the result more chroma-laden and then go back to linear sRGB from linear Rec 2020.
			float3 linear_sRGB = AgX_OUTSET_REC2020_TO_sRGB_MATRIX * linear_rec2020;
			linear_sRGB = fmaxf(linear_sRGB, make_float3(0.0f));

			// sRGB IEC 61966-2-1 2.2 Exponent Reference EOTF Display
			float3 non_linear_sRGB = RGBSpectrum(linear_sRGB).linearTosRGB().toFloat3();

			return non_linear_sRGB;
		}
	}

	//FILM============================================================================================
		//Not needed but added for parity with PBRTv4 implementation
	class Film
	{
	public:
		__device__ float3 computeNormalizedNonLinearSRGB(const RGBSpectrum& linear_srgb_radiance) const
		{
			float3 linear_rec_709 = linear_srgb_radiance.toFloat3();//sRGB and rec 709 have exactly same primaries but different gamma
			float3 display_color = AgXMinimal::AgXFitted(linear_rec_709, white_point_ev, black_point_ev);
			display_color = AgXMinimal::AgXLook(display_color);
			display_color = AgXMinimal::AgXFittedEOTF(display_color);
			//NOTE: display_color in NOT sRGB; its Rec. 709 with gamma 2.2(unlike usual 2.4); highly similar, different OETF however
			return display_color;
		}

		float luminance_exposure_scalar = 1.0f;//TODO: fix this retarded shit
		float black_point_ev = -10.0f;
		float white_point_ev = 6.5f;
	};
}