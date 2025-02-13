#pragma once
#include "containers.cuh"
#include "interaction.cuh"
#include "color.cuh"

namespace KittlesPT
{
	struct MaterialEvalContext;

	struct TextureEvalContext
	{
		__device__ explicit TextureEvalContext(const SurfaceInteraction& surf) :
			wpos(surf.world_position), uv(surf.uv)
		{}

		__device__ explicit TextureEvalContext(const MaterialEvalContext& ctx);

		__device__ TextureEvalContext(float3 wpos, float2 uv) :
			wpos(wpos), uv(uv)
		{};
		float3 wpos;
		float2 uv;
	};
	//TODO: decode from srgb before using
	class Texture
	{
	public:
		Texture(int width, int height, int channel_count, int bit_depth, int pixel_buffer_index) :
			width(width), height(height),
			channel_count(channel_count), bit_depth(bit_depth),
			pixel_buffer_index(pixel_buffer_index) {}

		//implied to be using UV mapping; returns normalized(0=>1)
		__device__ RGBSpectrum evaluate(const ShaderData& shader_data, const TextureEvalContext& ctx)
		{
			if (pixel_buffer_index < 0 || pixel_buffer_index >= shader_data.pixel_buffer.num)
			{
				return RGBSpectrum(0.0f);
			}

			size_t px_stride = channel_count * (bit_depth / 8); // `bit_depth` is a guarenteed multiple of 8
			size_t row_stride = width * px_stride;

			float2 st = make_float2(ctx.uv.x - floorf(ctx.uv.x), ctx.uv.y - floorf(ctx.uv.y));
			int32_t x = int32_t(st.x * width) % width;
			int32_t y = int32_t(st.y * height) % height;

			if (x < 0) x += width;
			if (y < 0) y += height;

			size_t px_offset = y * row_stride + x * px_stride;

			if (px_offset + 3 >= shader_data.pixel_buffer.num)
			{
				return RGBSpectrum(0.0f);
			}

			uint8_t* px_buff = &shader_data.pixel_buffer.data[pixel_buffer_index];
			uint8_t* triplet_buff = &px_buff[px_offset];

			return RGBSpectrum(
				triplet_buff[0] * Constants::INV_255,
				triplet_buff[1] * Constants::INV_255,
				triplet_buff[2] * Constants::INV_255
			);
		}

	public:
		int width = 0, height = 0;
		int channel_count = 3, bit_depth = 8;
		int pixel_buffer_index = -1;
	};
}/*KittlesPT*/