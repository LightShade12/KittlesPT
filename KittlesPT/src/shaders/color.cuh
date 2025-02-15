#pragma once
#include "maths/vector_maths.cuh"
#include <cuda_runtime.h>

namespace KittlesPT
{
	/*
	* Color Space conversions
	* Courtesy:https://github.com/Jessie-LC
	* From: https://github.com/Jessie-LC/open-source-utility-code
	*/

	inline __device__ float3 convertRGB2XYZ(float3 _rgb)
	{
		// Reference(s):
		// - RGB/XYZ Matrices
		//   https://web.archive.org/web/20191027010220/http://www.brucelindbloom.com/index.html?Eqn_RGB_XYZ_Matrix.html
		float3 xyz;
		xyz.x = dot(make_float3(0.4124564, 0.3575761, 0.1804375), _rgb);
		xyz.y = dot(make_float3(0.2126729, 0.7151522, 0.0721750), _rgb);
		xyz.z = dot(make_float3(0.0193339, 0.1191920, 0.9503041), _rgb);
		return xyz;
	}

	inline __device__ float3 convertXYZ2RGB(float3 _xyz)
	{
		float3 rgb;
		rgb.x = dot(make_float3(3.2404542, -1.5371385, -0.4985314), _xyz);
		rgb.y = dot(make_float3(-0.9692660, 1.8760108, 0.0415560), _xyz);
		rgb.z = dot(make_float3(0.0556434, -0.2040259, 1.0572252), _xyz);
		return rgb;
	}
	inline __device__ float3 convertXYZ2Yxy(float3 _xyz)
	{
		// Reference(s):
		// - XYZ to xyY
		//   https://web.archive.org/web/20191027010144/http://www.brucelindbloom.com/index.html?Eqn_XYZ_to_xyY.html
		float inv = 1.0 / dot(_xyz, make_float3(1.0, 1.0, 1.0));
		return make_float3(_xyz.y, _xyz.x * inv, _xyz.y * inv);
	}
	inline __device__ float3 convertYxy2XYZ(float3 _Yxy)
	{
		// Reference(s):
		// - xyY to XYZ
		//   https://web.archive.org/web/20191027010036/http://www.brucelindbloom.com/index.html?Eqn_xyY_to_XYZ.html
		float3 xyz;
		xyz.x = _Yxy.x * _Yxy.y / _Yxy.z;
		xyz.y = _Yxy.x;
		xyz.z = _Yxy.x * (1.0 - _Yxy.y - _Yxy.z) / _Yxy.z;
		return xyz;
	}
	inline __device__ float3 convertRGB2Yxy(float3 _rgb)
	{
		return convertXYZ2Yxy(convertRGB2XYZ(_rgb));
	}

	inline __device__ float3 convertYxy2RGB(float3 _Yxy)
	{
		return convertXYZ2RGB(convertYxy2XYZ(_Yxy));
	}

	inline __device__ float3 plancks(float t, float3 lambda) {
		const float h = 6.63e-16;
		const float c = 3.0e17;
		const float k = 1.38e-5;
		float3 p1 = (2.0 * h * pow(c, 2.0)) / powf(lambda, make_float3(5.0));
		float3 p2 = expf(h * c / (lambda * k * t)) - make_float3(1.0f);
		return (p1 / p2) * pow(1e9, 2.0);
	}

	inline __device__ float3 blackbody(float t) {
		float3 rgb = plancks(t, make_float3(660.0, 550.0, 440.0));
		rgb = rgb / max(rgb.x, max(rgb.y, rgb.z));

		return rgb;
	}

	//---------------------

	// Rec. 709 luminance coefficients for linear RGB
	__constant__ constexpr float3 rec709_luminance_coeffs{ 0.2126f,0.7152f,0.0722f };

	class RGBSpectrum
	{
	public:

		//================================================================================================================
		//CONSTRUCTORS
		//================================================================================================================
		__device__ RGBSpectrum(float r, float g, float b) :
			r(r), g(g), b(b) {};

		__device__ explicit RGBSpectrum(float v) :
			r(v), g(v), b(v) {};

		__device__ explicit RGBSpectrum(float3 v) :
			r(v.x), g(v.y), b(v.z) {};

		__device__ explicit RGBSpectrum(float4 v) :
			r(v.x), g(v.y), b(v.z) {};

		__device__ float3 toFloat3() const
		{
			return make_float3(r, g, b);
		}

		//================================================================================================================
		//OPERATORS
		//================================================================================================================

		//BOOLEAN-----------------------------------------------------
		__device__ bool operator!() const
		{
			return (r == 0.f && g == 0.f && b == 0.f);
		}

		__device__ operator bool() const
		{
			return (r != 0.f || g != 0.f || b != 0.f);
		}

		__device__ bool operator==(const RGBSpectrum& s) const
		{
			return r == s.r && g == s.g && b == s.b;
		}

		__device__ bool operator!=(const RGBSpectrum& s) const
		{
			return r != s.r || g != s.g || b != s.b;
		}

		//ARITHMETIC-------------------------------------------------

		__device__ RGBSpectrum operator-() const
		{
			return RGBSpectrum(-r, -g, -b);
		}

		//ADDITION--------------
		__device__ RGBSpectrum& operator+=(const RGBSpectrum& s)
		{
			r += s.r;
			g += s.g;
			b += s.b;
			return *this;
		}

		__device__ RGBSpectrum operator+(float s) const
		{
			return RGBSpectrum(r + s, g + s, b + s);
		}

		__device__ RGBSpectrum operator+(const RGBSpectrum& s) const
		{
			return RGBSpectrum(r + s.r, g + s.g, b + s.b);
		}

		//SUBTRACTION--------------
		__device__ RGBSpectrum& operator-=(const RGBSpectrum& s)
		{
			r -= s.r;
			g -= s.g;
			b -= s.b;
			return *this;
		}

		__device__ RGBSpectrum operator-(float s) const
		{
			return RGBSpectrum(r - s, g - s, b - s);
		}

		__device__ RGBSpectrum operator-(const RGBSpectrum& s) const
		{
			return RGBSpectrum(r - s.r, g - s.g, b - s.b);
		}

		//MULTIPLICATION--------------

		__device__ RGBSpectrum& operator*=(float a)
		{
			r *= a;
			g *= a;
			b *= a;
			return *this;
		}
		__device__ RGBSpectrum& operator*=(const RGBSpectrum& s)
		{
			r *= s.r;
			g *= s.g;
			b *= s.b;
			return *this;
		}

		__device__ RGBSpectrum operator*(float a) const
		{
			return RGBSpectrum(a * r, a * g, a * b);
		}
		__device__ RGBSpectrum operator*(const RGBSpectrum& s) const
		{
			return RGBSpectrum(r * s.r, g * s.g, b * s.b);
		}

		//DIVISION--------------

		__device__ RGBSpectrum& operator/=(float a)
		{
			r /= a;
			g /= a;
			b /= a;
			return *this;
		}

		__device__ RGBSpectrum& operator/=(const RGBSpectrum& s)
		{
			r /= s.r;
			g /= s.g;
			b /= s.b;
			return *this;
		}

		__device__ RGBSpectrum operator/(float a) const
		{
			return RGBSpectrum(r / a, g / a, b / a);
		}

		__device__ RGBSpectrum operator/(const RGBSpectrum& s) const
		{
			return RGBSpectrum(r / s.r, g / s.g, b / s.b);
		}

		//================================================================================================================
		//UTILITIES
		//================================================================================================================

		//pmax val=1000
		__device__ RGBSpectrum clampOutput()
		{
			return RGBSpectrum(KittlesPT::clampOutput(toFloat3()));
		}

		__device__ float maxComponentValue()
		{
			return fmaxf(r, fmaxf(g, b));
		}

		__device__ float Average() const
		{
			return (r + g + b) / 3.0f;
		}

		//for readonly
		__device__ float operator[](int c) const
		{
			if (c == 0)
				return r;
			else if (c == 1)
				return g;
			return b;
		}

		//for assignment
		__device__ float& operator[](int c)
		{
			if (c == 0)
				return r;
			else if (c == 1)
				return g;
			return b;
		}

		//================================================================================================================
		//COLOR SPACE OPERATIONS
		//================================================================================================================

		//Y value
		__device__ float getLuminance() const
		{
			return (rec709_luminance_coeffs.x * r) + (rec709_luminance_coeffs.y * g) + (rec709_luminance_coeffs.z * b);
		}

		__device__ float Y() const
		{
			return getLuminance();
		}

		__device__ static RGBSpectrum fromXYZ(float3 v)
		{
			return RGBSpectrum(convertXYZ2RGB(v));
		}

		__device__ static RGBSpectrum fromYxy(float3 v)
		{
			return RGBSpectrum(convertYxy2RGB(v));
		}

		__device__ float3 toXYZ() const
		{
			return convertRGB2XYZ(toFloat3());
		}

		__device__ float3 toYxy() const
		{
			return convertXYZ2Yxy(toXYZ());
		}

		__device__ RGBSpectrum linearToGamma2_2() const {
			const float i = 1.0f / 2.2f;
			return RGBSpectrum(powf(r, i), powf(g, i), powf(b, i));
		}

		__device__ RGBSpectrum gamma2_2ToLinear() const {
			const float i = 2.2f;
			return RGBSpectrum(powf(r, i), powf(g, i), powf(b, i));
		}

		__device__ RGBSpectrum linearToGamma2_4() const {
			const float i = 1.0f / 2.4f;
			return RGBSpectrum(powf(r, i), powf(g, i), powf(b, i));
		}

		__device__ RGBSpectrum gamma2_4ToLinear() const {
			const float i = 2.4f;
			return RGBSpectrum(powf(r, i), powf(g, i), powf(b, i));
		}

		//Same primaries and white-point as ITU-R BT.709
		//IEC 61966-2-1:1999

		__device__ float sRGBEncoding(float v) {
			constexpr float V = 0.0031308f;
			constexpr float A = 12.92f;
			constexpr float C = 0.055f;
			constexpr float T = 2.4f;

			return (v <= V) ? (A * v) : ((1 + C) * powf(v, 1.0f / T) - C);
		}

		__device__ float sRGBDecoding(float v) {
			constexpr float U = 0.04045f;
			constexpr float A = 12.92f;
			constexpr float C = 0.055f;
			constexpr float T = 2.4f;

			return (v <= U) ? (v / A) : powf((v + C) / (1 + C), T);
		}

		// linear RGB to sRGB (normalized [0,1])
		__device__ RGBSpectrum linearTosRGB() {
			return RGBSpectrum(sRGBEncoding(r), sRGBEncoding(g), sRGBEncoding(b));
		}

		// sRGB to linear RGB (normalized [0,1])
		__device__ RGBSpectrum sRGBToLinear() {
			return RGBSpectrum(sRGBDecoding(r), sRGBDecoding(g), sRGBDecoding(b));
		}

	public:

		float r = 0.0f, g = 0.0f, b = 0.0f;
	};

	//OPERATORS==========================================================================

	inline __device__ RGBSpectrum operator-(float b, RGBSpectrum a)
	{
		return RGBSpectrum(b - a.r, b - a.g, b - a.b);
	}

	inline __device__ RGBSpectrum operator+(float b, RGBSpectrum a)
	{
		return RGBSpectrum(b + a.r, b + a.g, b + a.b);
	};
	inline __device__ RGBSpectrum operator*(float b, RGBSpectrum a)
	{
		return RGBSpectrum(b * a.r, b * a.g, b * a.b);
	};
	inline __device__ RGBSpectrum operator/(float b, RGBSpectrum a)
	{
		return RGBSpectrum(b / a.r, b / a.g, b / a.b);
	};
}/*KittlesPT*/

inline __device__ KittlesPT::RGBSpectrum lerp(KittlesPT::RGBSpectrum a, KittlesPT::RGBSpectrum b, float t)
{
	return a + t * (b - a);
}

inline __device__ KittlesPT::RGBSpectrum clamp(KittlesPT::RGBSpectrum v, KittlesPT::RGBSpectrum a, KittlesPT::RGBSpectrum b)
{
	return KittlesPT::RGBSpectrum(clamp(v.r, a.r, b.r), clamp(v.g, a.g, b.g), clamp(v.b, a.b, b.b));
}

inline __device__ KittlesPT::RGBSpectrum powf(KittlesPT::RGBSpectrum x, float y)
{
	return KittlesPT::RGBSpectrum(powf(x.r, y), powf(x.g, y), powf(x.b, y));
}

inline __device__ KittlesPT::RGBSpectrum  exp(KittlesPT::RGBSpectrum  x)
{
	return KittlesPT::RGBSpectrum(expf(x.r), expf(x.g), expf(x.b));
}