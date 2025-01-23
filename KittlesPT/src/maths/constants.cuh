#pragma once
#include <cuda_runtime.h>

namespace KittlesPT
{
	namespace Constants
	{
		__constant__ constexpr float RADS_PER_DEG = 0.01745329251f;
		__constant__ constexpr float PI = 3.14159265358979323846f;
		__constant__ constexpr float INV_PI = 0.31830988618379067154f;
		__constant__ constexpr float INV_2PI = 0.15915494309189533577f;
		__constant__ constexpr float INV_4PI = 0.07957747154594766788f;
		__constant__ constexpr float PI_OVER_2 = 1.57079632679489661923f;
		__constant__ constexpr float PI_OVER_4 = 0.78539816339744830961f;
		__constant__ constexpr float SQRT2 = 1.41421356237309504880f;
		__constant__ constexpr float INV_UINT32MAX = 1.0f / float(UINT32_MAX);
		__constant__ constexpr float INV_255 = 0.00392156862f;

		__constant__ constexpr float TRIANGLE_INTERSECTION_EPSILON = 0.000001f;
		__constant__ constexpr float HIT_EPSILON = 0.001f;
		__constant__ constexpr float GGX_ROUGHNESS_EPSILON = 0.045f;//TODO:should these be here?
		__constant__ constexpr int ASVGF_STRATUM_SIZE = 3;
		__constant__ constexpr float HISTOGRAM_LUMINANCE_EPSILON = 0.005f;
		__constant__ constexpr int HISTOGRAM_SIZE = 256;
	}
}