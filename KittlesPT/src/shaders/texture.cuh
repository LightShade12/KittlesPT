#pragma once
#include <vector_types.h>

namespace KittlesPT
{
	class RGBSpectrum;
	struct GlobalShaderData;
	struct SurfaceInteraction;

	struct TextureEvalContext
	{
		__device__ TextureEvalContext(const SurfaceInteraction& surf);
		float3 wpos;
		float2 uv;
	};

	class Texture
	{
	public:
		Texture(int width, int height, int channel_count, int bit_depth, int pixel_buffer_index) :
			width(width), height(height),
			channel_count(channel_count), bit_depth(bit_depth),
			pixel_buffer_index(pixel_buffer_index) {}

		//implied to be using UV mapping
		__device__ RGBSpectrum evaluate(const GlobalShaderData& shader_data, TextureEvalContext ctx);

	public:
		int width = 0, height = 0;
		int channel_count = 3, bit_depth = 8;
		int pixel_buffer_index = -1;
	};
}