#include "texture.cuh"
#include "color.cuh"
#include "interaction.cuh"
#include "material.cuh"
#include "containers.cuh"

namespace KittlesPT
{
	//TODO: replace with CUDA texture memory
	__device__ TextureEvalContext::TextureEvalContext(const SurfaceInteraction& surf) :
		wpos(surf.world_position), uv(surf.uv)
	{}

	__device__ TextureEvalContext::TextureEvalContext(const MaterialEvalContext& ctx) :
		uv(ctx.uv), wpos(ctx.wpos)
	{}

	//=================================================================================================================

	__device__ RGBSpectrum Texture::evaluate(const ShaderData& shader_data, const TextureEvalContext& ctx)
	{
		if (pixel_buffer_index < 0 || pixel_buffer_index >= shader_data.pixel_buffer.num)
		{
			return RGBSpectrum(0.0f);
		}

		size_t px_stride = channel_count * (bit_depth / 8); // `bit_depth` is a guarenteed multiple of 8
		size_t row_stride = width * px_stride;

		float2 st = make_float2(ctx.uv.x - floorf(ctx.uv.x), ctx.uv.y - floorf(ctx.uv.y));
		int x = int(st.x * width) % width;
		int y = int(st.y * height) % height;

		if (x < 0) x += width;
		if (y < 0) y += height;

		size_t px_offset = y * row_stride + x * px_stride;

		if (px_offset + 3 >= shader_data.pixel_buffer.num)
		{
			return RGBSpectrum(0.0f);
		}

		unsigned char* px_buff = &shader_data.pixel_buffer.data[pixel_buffer_index];
		unsigned char* triplet_buff = &px_buff[px_offset];

		return RGBSpectrum(
			triplet_buff[0] * Constants::INV_255,
			triplet_buff[1] * Constants::INV_255,
			triplet_buff[2] * Constants::INV_255
		);
	}
}/*KittlesPT*/