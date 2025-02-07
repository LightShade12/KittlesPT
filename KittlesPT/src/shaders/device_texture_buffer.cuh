#pragma once
#include "error_check.cuh"
#include "maths/vector_maths.cuh"

#include <cuda.h>
//#define __CUDACC__
//#include <device_functions.h>
#include <cuda_runtime_api.h>//should be used instead of device_functions.h
#include <glad/glad.h>
#include <cuda_gl_interop.h>

#include <iostream>

namespace KittlesPT
{
	//TODO: rename this
	struct DeviceTextureBuffer
	{
		inline __device__ void textureWrite(float4 value, int2 pixel_coord) const
		{
			surf2Dwrite<float4>(value, m_surface_object, pixel_coord.x * (int)sizeof(float4), pixel_coord.y, cudaBoundaryModeZero);
		}

		inline __device__ void textureWriteUV(float4 value, float2 uv_coord) const
		{
			int2 pixel_coord = make_int2(uv_coord * dimensions);
			surf2Dwrite<float4>(value, m_surface_object, pixel_coord.x * (int)sizeof(float4), pixel_coord.y, cudaBoundaryModeZero);
		}

		inline __device__ float4 textureReadNearest(float2 pixel_coord) const
		{
			return surf2Dread<float4>(m_surface_object, int(pixel_coord.x) * (int)sizeof(float4), int(pixel_coord.y), cudaBoundaryModeClamp);
		}

		inline __device__ float4 textureReadNearestUV(float2 uv_coord) const
		{
			int2 pixel_coord = make_int2(uv_coord * dimensions);
			//pixel_coord = clamp(pixel_coord, make_int2(0, 0), dimensions - 1);
			return surf2Dread<float4>(m_surface_object, pixel_coord.x * (int)sizeof(float4), pixel_coord.y, cudaBoundaryModeClamp);
		}

		inline __device__ float4 textureReadBilinear(float2 pixel_coord, float filter_alpha) const
		{
			/*
			* Coordinates(verified with uv shading):
			* +1
			* ^
			* ||
			* 0 ==> +1
			*/

			//TODO:consider half pixel for centre sampling

			int2 discrete = make_int2(pixel_coord);

			//tap coord components
			int s0 = discrete.x;
			int s1 = discrete.x + 1;
			int t0 = discrete.y;
			int t1 = discrete.y + 1;

			float ws = pixel_coord.x - discrete.x;
			float wt = pixel_coord.y - discrete.y;

			float4 cp0 = textureReadNearest(make_float2(s0, t0));
			float4 cp1 = textureReadNearest(make_float2(s1, t0));
			float4 cp2 = textureReadNearest(make_float2(s0, t1));
			float4 cp3 = textureReadNearest(make_float2(s1, t1));

			float4 tc0 = lerp(cp0, cp1, ws);
			float4 tc1 = lerp(cp2, cp3, ws);
			float4 fc = lerp(tc0, tc1, wt);

			// point sampling for alpha
			if (!filter_alpha) {
				fc.w = cp0.w;
			}

			return fc;
		}

		inline __device__ float4 textureReadBilinearUV(float2 uv_coord, float filter_alpha) const
		{
			float2 pixel_coord = uv_coord * dimensions;
			return textureReadBilinear(pixel_coord, filter_alpha);
		}

		int2 dimensions;
		cudaSurfaceObject_t m_surface_object;
	};
}/*KittlesPT*/