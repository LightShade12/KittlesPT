#pragma once

#include <vector_types.h>
#include <cstdint>

namespace KittlesPT
{
	struct SurfaceInteraction;
	class RGBSpectrum;

	//Only works with normalized floats[0=>1]
	__device__ uint32_t floatToByte(float v);

	__device__ float byteToFloat(uint32_t v);

	//unused 8 bits MSB
	__device__ uint32_t float3ToUint(float3 v);

	__device__ float3 uintToFloat3(uint32_t v);

	__device__ uint32_t float4ToUint(float4 v);

	__device__ float4 uintToFloat4(uint32_t v);

	//returns uint32_t
	__device__ uint32_t packHalf2x16(float2 v);

	__device__ float2 unpackHalf2x16(uint32_t v);

	__device__ uint32_t floatBitsToUint(float v);

	__device__ float uintBitsToFloat(uint32_t v);

	__device__ uint4 float4BitsToUint4(float4 v);

	__device__ float4 uint4BitsToFloat4(uint4 v);

	//only used for tangent space normals
	__device__ float3 float2ToNormal(float2 nxy);

	/*
	* Unit Vector Packing
	* Courtesy:https://github.com/Jessie-LC
	* From: https://github.com/Jessie-LC/open-source-utility-code
	*/

	// Octahedral Unit Vector encoding
	// Intuitive, fast, and has very little error.
	__device__ float2 encodeUnitVector(float3 vector);
	__device__ float3 decodeUnitVector(float2 encoded);

	struct GBuffer
	{
		GBuffer() = default;
		__device__ GBuffer(const RGBSpectrum& albedo, const SurfaceInteraction& surf);
		float3 albedo{};
		float3 wgnorm{};
		float depth = INFINITY;

		__device__ float4 packGBuffer();

		__device__ static GBuffer unpackGBuffer(float4 data);
		/*
		* float3 wpos;
		* int primitive_id = -1;
		* float3 viewdir;
		* PBRT:
		* dzdx, dzdy
		* uv
		* wsnorm
		* variance estimates
		*/
	};
}/*KittlesPT*/