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
		//wrapping coords
		float2 st = make_float2(fracf(ctx.uv.x), fracf(ctx.uv.y));

		if (pixel_buffer_index < 0 || pixel_buffer_index >= shader_data.pixel_buffer.num)
		{
			return RGBSpectrum(0);
		}

		unsigned char* px_buff = &shader_data.pixel_buffer.data[pixel_buffer_index];
		size_t px_stride = channel_count * (bit_depth / 8.0f);//bit_depth is guarenteed to be 8-multiple
		size_t px_offset = int(st.x * width * px_stride) + int((st.y * height) * width * px_stride);

		if (px_offset + 3u >= shader_data.pixel_buffer.num)
		{
			return RGBSpectrum(0);
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