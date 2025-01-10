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

	__device__ float3 convertRGB2XYZ(float3 _rgb);

	__device__ float3 convertXYZ2RGB(float3 _xyz);

	__device__ float3 convertXYZ2Yxy(float3 _xyz);

	__device__ float3 convertYxy2XYZ(float3 _Yxy);

	__device__ float3 convertRGB2Yxy(float3 _rgb);

	__device__ float3 convertYxy2RGB(float3 _Yxy);

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

		//max val=1000
		__device__ RGBSpectrum clampOutput();

		__device__  float maxComponentValue()
		{
			return fmaxf(r, fmaxf(g, b));
		}

		__device__ float Average() const
		{
			return (r + g + b) / 3;
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
		__device__ float getLuminance()
		{
			// Rec. 709 luminance coefficients for linear RGB
			return (0.2126f * r) + (0.7152f * g) + (0.0722f * b);
		}

		__device__ float Y()
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

		__device__ float3 toXYZ()
		{
			return convertRGB2XYZ(toFloat3());
		}

		__device__ float3 toYxy()
		{
			return convertXYZ2Yxy(toXYZ());
		}

	public:

		float r = 0, g = 0, b = 0;
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

	inline __device__ RGBSpectrum lerp(RGBSpectrum a, RGBSpectrum b, float t)
	{
		return a + t * (b - a);
	}
}