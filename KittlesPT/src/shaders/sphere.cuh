#pragma once

#include "light.cuh"
#define __CUDACC__
#include <vector_types.h>
#include <math_functions.h>
#include <cuda_fp16.h>

namespace KittlesPT
{
	struct GlobalShaderData;
	class RGBSpectrum;
	class Ray;
	class BSDF;

	struct SurfaceInteraction
	{
		__device__ RGBSpectrum Le(const GlobalShaderData& shader_data, const Ray& ray) const;

		__device__ BSDF getBSDF(const GlobalShaderData& shader_data) const;

		__device__ Ray spawnRay(float3 wi, int scatter_flags) const;

		__device__ Ray spawnRayTo(float3 target) const;

		//--------------------------------------------------
		float distance = -1;
		float2 uv;
		float3 world_position;
		float3 world_geometric_normal;
		int material_id = -1;
		bool backface = false;
		const Light* light = nullptr;
	};

	struct Intersection
	{
		//closest hit shader
		__device__ SurfaceInteraction getSurfaceInteraction(const GlobalShaderData& shader_data, const Ray& ray);

		__device__ bool operator ! ();

		//--------------------------------------------------------
		float distance = -1;
		int instance_id = -1;
	};

	struct ShapeSample
	{
		ShapeSample() = default;

		__device__ ShapeSample(float3 wpos, float3 gwnorm, float pdf) :
			wpos(wpos), wgnorm(gwnorm), pdf(pdf) {};

		float3 wgnorm{};
		float3 wpos{};
		float pdf = 0;
	};

	struct ShapeSampleContext
	{
		__device__ ShapeSampleContext(const SurfaceInteraction& surf) :
			wpos(surf.world_position), wgnorm(surf.world_geometric_normal)
		{};

		__device__ ShapeSampleContext(const LightSampleContext& ctx) :
			wpos(ctx.w_pos), wgnorm(ctx.wgnorm)
		{};

		float3 wpos;
		float3 wgnorm;
	};

	class Sphere
	{
	public:
		__device__ __host__ Sphere(float radius_, float3 pos, int material_id, int light_id)//TODO: make it host only
			:radius(radius_), world_position(pos), material_id(material_id), light_id(light_id) {};

		__device__ Intersection intersect(const Ray& ray, float tmax) const;

		__device__ ShapeSample sample(float2 u2) const;

		__device__ ShapeSample sample(float2 u2, ShapeSampleContext ctx) const;

		__host__ __device__ float getArea() const;

		__host__ __device__ float getProjectedArea() const;

	public:
		int material_id = -1;
		int light_id = -1;
		float3 world_position;
		float radius = 1;
	};

	//Only works with normalized floats[0=>1]
	__device__ unsigned floatToByte(float v)
	{
		v = clamp(v, 0.0f, 1.0f);
		return unsigned(v * 255.0f) & 0xFFu;
	};

	__device__ float byteToFloat(unsigned v)
	{
		return float(v & 0xFFu) / 255.0f;
	};

	//unused 8 bits MSB
	__device__ unsigned float3ToUint(float3 v)
	{
		unsigned o = floatToByte(v.x) | (floatToByte(v.y) << 8) | (floatToByte(v.z) << 16);
		return o;
	};

	__device__ float3 uintToFloat3(unsigned v)
	{
		float3 o;
		o.x = byteToFloat(v);
		o.y = byteToFloat(v >> 8);
		o.z = byteToFloat(v >> 16);
		return o;
	};

	__device__ unsigned float4ToUint(float4 v)
	{
		unsigned o = floatToByte(v.x) | (floatToByte(v.y) << 8) | (floatToByte(v.z) << 16) | (floatToByte(v.w) << 24);
		return o;
	};

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
		return (unsigned)__internal_float_as_uint(v);
	};
	__device__ float uintBitsToFloat(unsigned v)
	{
		return (float)__internal_uint_as_float(v);
	};

	__device__ uint4 floatBitsToUint(float4 v)
	{
		return make_uint4(floatBitsToUint(v.x), floatBitsToUint(v.y),
			floatBitsToUint(v.z), floatBitsToUint(v.w));
	};

	__device__ float4 uintBitsToFloat(uint4 v)
	{
		return make_float4(uintBitsToFloat(v.x), uintBitsToFloat(v.y),
			uintBitsToFloat(v.z), uintBitsToFloat(v.w));
	};

	__device__ float3 float2ToNormal(float2 nxy)
	{
		float z = sqrtf(1.0 - dot(nxy, nxy));
		return make_float3(nxy.x, nxy.y, z);
	};

	struct NDIBuffer
	{
		float3 wgnorm;
		float depth;
		int instance_id = -1;
	};

	struct GBuffer
	{
		float3 wpos;
		float3 wgnorm;
		float depth;
		int instance_id = -1;
		float3 albedo;
		float3 viewdir;
		/*
		* PBRT:
		* dzdx, dzdy
		* uv
		* wsnorm
		* variance estimates
		*/
	};

	float4 packGBuffer(GBuffer gb)
	{
		uint4 v{};
		float3 wgnorm = (gb.wgnorm + 1.0f) * 0.5, viewdir = (gb.viewdir + 1.0f) * 0.5;
		v.x = float4ToUint(make_float4(wgnorm.x, wgnorm.y, viewdir.x, viewdir.y));
		v.y = float4ToUint(make_float4(gb.albedo, gb.depth));//depth is linearized and [0->1]
		v.z = gb.instance_id;
		//v.w = 4 bytes unused
		return uintBitsToFloat(v);
	}

	GBuffer unpackGBuffer(float4 data)
	{
		GBuffer gb;
		uint4 v = floatBitsToUint(data);
		float4 nxy_vxy = uintToFloat4(v.x);
		gb.wgnorm = float2ToNormal(normalize(make_float2(nxy_vxy.x, nxy_vxy.y) * 2.0f - 1.0f));
		gb.viewdir = float2ToNormal(normalize(make_float2(nxy_vxy.z, nxy_vxy.w) * 2.0f - 1.0f));
		float4 albedodepth = uintToFloat4(v.y);
		gb.albedo = make_float3(albedodepth);
		gb.depth = albedodepth.w;
		gb.instance_id = v.z;
		gb.wpos = gb.viewdir * gb.depth;

		return gb;
	}
}