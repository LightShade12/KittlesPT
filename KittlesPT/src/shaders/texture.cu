#include "texture.cuh"
#include "color.cuh"
#include "sphere.cuh"
#include "containers.cuh"

namespace KittlesPT
{
	__device__ TextureEvalContext::TextureEvalContext(const SurfaceInteraction& surf) :
		wpos(surf.world_position), uv(surf.uv)
	{}

	__device__ RGBSpectrum Texture::evaluate(const GlobalShaderData& shader_data, TextureEvalContext ctx)
	{
		float2 st = make_float2(ctx.uv.x - floorf(ctx.uv.x), ctx.uv.y - floorf(ctx.uv.y));

		if (pixel_buffer_index < 0 || pixel_buffer_index >= shader_data.pixel_buffer.num)
		{
			return RGBSpectrum(0.0f);
		}

		unsigned char* px_buff = &shader_data.pixel_buffer.data[pixel_buffer_index];
		size_t px_stride = channel_count * (bit_depth / 8); // `bit_depth` must be a multiple of 8
		size_t row_stride = width * px_stride;

		int x = int(st.x * width) % width;
		int y = int(st.y * height) % height;

		if (x < 0) x += width;
		if (y < 0) y += height;

		size_t px_offset = y * row_stride + x * px_stride;

		if (px_offset + 3 >= shader_data.pixel_buffer.num)
		{
			return RGBSpectrum(0.0f);
		}

		unsigned char* triplet_buff = &px_buff[px_offset];

		RGBSpectrum rgb = RGBSpectrum(
			triplet_buff[0] / 255.0f,
			triplet_buff[1] / 255.0f,
			triplet_buff[2] / 255.0f
		);

		return rgb;
	}
}