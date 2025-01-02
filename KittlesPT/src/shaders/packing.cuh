
#define __CUDACC__
#include <cuda.h>
#include <cuda_fp16.h>

namespace KittlesPT
{
	////Only works with normalized floats[0=>1]
	//__device__ unsigned floatToByte(float v)
	//{
	//	v = clamp(v, 0.0f, 1.0f);
	//	return unsigned(v * 255.0f) & 0xFFu;
	//};

	//__device__ float byteToFloat(unsigned v)
	//{
	//	return float(v & 0xFFu) / 255.0f;
	//};

	////unused 8 bits MSB
	//__device__ unsigned float3ToUint(float3 v)
	//{
	//	unsigned o = floatToByte(v.x) | (floatToByte(v.y) << 8) | (floatToByte(v.z) << 16);
	//	return o;
	//};

	//__device__ float3 uintToFloat3(unsigned v)
	//{
	//	float3 o;
	//	o.x = byteToFloat(v);
	//	o.y = byteToFloat(v >> 8);
	//	o.z = byteToFloat(v >> 16);
	//	return o;
	//};

	//__device__ unsigned float4ToUint(float4 v)
	//{
	//	unsigned o = floatToByte(v.x) | (floatToByte(v.y) << 8) | (floatToByte(v.z) << 16) | (floatToByte(v.w) << 24);
	//	return o;
	//};

	//__device__ float4 uintToFloat4(unsigned v)
	//{
	//	float4 o;
	//	o.x = byteToFloat(v);
	//	o.y = byteToFloat(v >> 8);
	//	o.z = byteToFloat(v >> 16);
	//	o.w = byteToFloat(v >> 24);
	//	return o;
	//};

	////returns unsigned
	//__device__ unsigned packHalf2x16(float2 v)
	//{
	//	unsigned short x = __half_as_ushort(__float2half(v.x));
	//	unsigned short y = __half_as_ushort(__float2half(v.y));

	//	return (unsigned(y) << 16) | unsigned(x);
	//};

	//__device__ float2 unpackHalf2x16(unsigned v)
	//{
	//	half x = __ushort_as_half(v & 0xFFFFu);
	//	half y = __ushort_as_half((v >> 16) & 0xFFFFu);

	//	return make_float2(__half2float(x), __half2float(y));
	//};

	//__device__ unsigned floatBitsToUint(float v)
	//{
	//	return (unsigned)__internal_float_as_uint(v);
	//};
	//__device__ float uintBitsToFloat(unsigned v)
	//{
	//	return (float)__internal_uint_as_float(v);
	//};

	//__device__ uint4 float4BitsToUint4(float4 v)
	//{
	//	return make_uint4(floatBitsToUint(v.x), floatBitsToUint(v.y),
	//		floatBitsToUint(v.z), floatBitsToUint(v.w));
	//};

	//__device__ float4 uint4BitsToFloat4(uint4 v)
	//{
	//	return make_float4(uintBitsToFloat(v.x), uintBitsToFloat(v.y),
	//		uintBitsToFloat(v.z), uintBitsToFloat(v.w));
	//};

	//__device__ float3 float2ToNormal(float2 nxy)
	//{
	//	float z = sqrtf(1.0 - dot(nxy, nxy));
	//	return make_float3(nxy.x, nxy.y, z);
	//};

	//struct NDIBuffer
	//{
	//	float3 wgnorm;
	//	float depth;
	//	int instance_id = -1;
	//};

	//struct GBuffer
	//{
	//	float3 wpos;
	//	float3 wgnorm;
	//	float depth;
	//	int instance_id = -1;
	//	float3 albedo;
	//	float3 viewdir;
	//	/*
	//	* PBRT:
	//	* dzdx, dzdy
	//	* uv
	//	* wsnorm
	//	* variance estimates
	//	*/
	//};

	//__device__ float4 packGBuffer(GBuffer gb)
	//{
	//	uint4 v{};
	//	float3 wgnorm = (gb.wgnorm + 1.0f) * 0.5, viewdir = (gb.viewdir + 1.0f) * 0.5;
	//	v.x = float4ToUint(make_float4(wgnorm.x, wgnorm.y, viewdir.x, viewdir.y));
	//	v.y = float4ToUint(make_float4(gb.albedo, gb.depth));//depth is linearized and [0->1]
	//	v.z = gb.instance_id;
	//	//v.w = 4 bytes unused
	//	return uint4BitsToFloat4(v);
	//}

	//__device__ GBuffer unpackGBuffer(float4 data)
	//{
	//	GBuffer gb;
	//	uint4 v = float4BitsToUint4(data);
	//	float4 nxy_vxy = uintToFloat4(v.x);
	//	gb.wgnorm = float2ToNormal(normalize(make_float2(nxy_vxy.x, nxy_vxy.y) * 2.0f - 1.0f));
	//	gb.viewdir = float2ToNormal(normalize(make_float2(nxy_vxy.z, nxy_vxy.w) * 2.0f - 1.0f));
	//	float4 albedodepth = uintToFloat4(v.y);
	//	gb.albedo = make_float3(albedodepth);
	//	gb.depth = albedodepth.w;
	//	gb.instance_id = v.z;
	//	gb.wpos = gb.viewdir * gb.depth;

	//	return gb;
	//}
}