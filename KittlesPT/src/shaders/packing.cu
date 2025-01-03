#include "packing.cuh"
#include "interaction.cuh"
#include "color.cuh"
#include "maths/linear_algebra.cuh"

#include <cuda.h>
#include <cuda_fp16.h>

namespace KittlesPT
{
	__device__ unsigned floatToByte(float v)
	{
		v = clamp(v, 0.0f, 1.0f);
		return unsigned(v * 255.0f) & 0xFFu;
	}
	__device__ float byteToFloat(unsigned v)
	{
		return float(v & 0xFFu) / 255.0f;
	}
	__device__ unsigned float3ToUint(float3 v)
	{
		unsigned o = floatToByte(v.x) | (floatToByte(v.y) << 8) | (floatToByte(v.z) << 16);
		return o;
	}
	__device__ float3 uintToFloat3(unsigned v)
	{
		float3 o;
		o.x = byteToFloat(v);
		o.y = byteToFloat(v >> 8);
		o.z = byteToFloat(v >> 16);
		return o;
	}
	__device__ unsigned float4ToUint(float4 v)
	{
		unsigned o = floatToByte(v.x) | (floatToByte(v.y) << 8) | (floatToByte(v.z) << 16) | (floatToByte(v.w) << 24);
		return o;
	}

	__device__ float4 uintToFloat4(unsigned v)
	{
		float4 o;
		o.x = byteToFloat(v);
		o.y = byteToFloat(v >> 8);
		o.z = byteToFloat(v >> 16);
		o.w = byteToFloat(v >> 24);
		return o;
	};

	//returns unsigned
	__device__ unsigned packHalf2x16(float2 v)
	{
		unsigned short x = __half_as_ushort(__float2half(v.x));
		unsigned short y = __half_as_ushort(__float2half(v.y));

		return (unsigned(y) << 16) | unsigned(x);
	};

	__device__ float2 unpackHalf2x16(unsigned v)
	{
		half x = __ushort_as_half(v & 0xFFFFu);
		half y = __ushort_as_half((v >> 16) & 0xFFFFu);

		return make_float2(__half2float(x), __half2float(y));
	};

	__device__ unsigned floatBitsToUint(float v)
	{
		return (unsigned)__float_as_uint(v);
	};
	__device__ float uintBitsToFloat(unsigned v)
	{
		return (float)__uint_as_float(v);
	};

	__device__ uint4 float4BitsToUint4(float4 v)
	{
		return make_uint4(floatBitsToUint(v.x), floatBitsToUint(v.y),
			floatBitsToUint(v.z), floatBitsToUint(v.w));
	};

	__device__ float4 uint4BitsToFloat4(uint4 v)
	{
		return make_float4(uintBitsToFloat(v.x), uintBitsToFloat(v.y),
			uintBitsToFloat(v.z), uintBitsToFloat(v.w));
	};

	__device__ float3 float2ToNormal(float2 nxy)
	{
		float z = sqrtf(1.0 - dot(nxy, nxy));
		return make_float3(nxy.x, nxy.y, z);
	};

	/*
	* Unit Vector Encoding & Decoding
	* Courtesy:https://github.com/Jessie-LC
	* From: https://github.com/Jessie-LC/open-source-utility-code
	*/

	// Octahedral Unit Vector encoding
	// Intuitive, fast, and has very little error.
	__device__ float2 encodeUnitVector(float3 vector)
	{
		// Scale down to octahedron, project onto XY plane
		float absX = abs(vector.x);
		float absY = abs(vector.y);
		float absZ = abs(vector.z);
		float sum = absX + absY + absZ;

		float encodedX = vector.x / sum;
		float encodedY = vector.y / sum;

		// Reflect -Z hemisphere folds over the diagonals
		if (vector.z <= 0.0f) {
			float reflectX = 1.0f - abs(encodedY);
			float reflectY = 1.0f - abs(encodedX);
			encodedX = encodedX >= 0.0f ? reflectX : -reflectX;
			encodedY = encodedY >= 0.0f ? reflectY : -reflectY;
		}

		return make_float2(encodedX, encodedY);
	}

	__device__ float3 decodeUnitVector(float2 encoded) {
		// Extract Z component
		float absX = abs(encoded.x);
		float absY = abs(encoded.y);
		float z = 1.0f - absX - absY;

		float3 vector = make_float3(encoded.x, encoded.y, z);

		// Reflect -Z hemisphere folds over the diagonals
		float t = max(-z, 0.0f);

		if (vector.x >= 0.0f) {
			vector.x -= t;
		}
		else {
			vector.x += t;
		}

		if (vector.y >= 0.0f) {
			vector.y -= t;
		}
		else {
			vector.y += t;
		}

		// Normalize and return
		return normalize(vector);
	}

	__device__ float4 packGBuffer(const GBuffer& gb)
	{
		uint4 out;
		float2 encoded_nrm = encodeUnitVector(gb.wgnorm);
		out.x = floatBitsToUint(encoded_nrm.x);
		out.y = floatBitsToUint(encoded_nrm.y);
		out.z = float3ToUint(gb.albedo);
		out.w = floatBitsToUint(gb.depth);

		return uint4BitsToFloat4(out);
	}

	__device__ GBuffer unpackGBuffer(float4 data)
	{
		GBuffer gb;
		uint4 in = float4BitsToUint4(data);
		gb.albedo = uintToFloat3(in.z);
		gb.wgnorm = decodeUnitVector(make_float2(uintBitsToFloat(in.x), uintBitsToFloat(in.y)));
		gb.depth = uintBitsToFloat(in.w);
		return gb;
	}

	//GBUFFER==========================================================================================================================

	__device__ GBuffer::GBuffer(const RGBSpectrum& albedo, const SurfaceInteraction& surf)
		:
		albedo(albedo.toFloat3()), wgnorm(surf.world_geometric_normal),
		depth(surf.distance)
	{}
}