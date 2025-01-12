#include "bloom.cuh"
#include "shading_kernel.cuh"

namespace KittlesPT
{
	__device__ float4 texRead36Texel(DeviceTextureBuffer t_tex, int2 t_res, float2 t_pixel_coord)
	{
		//TODO:karis average
		//top left
		float2 a_pix = clamp(t_pixel_coord + make_float2(-2, 2), make_float2(0), make_float2(t_res - 1));
		float4 a = t_tex.textureReadBilinear(a_pix, true);
		//top
		float2 b_pix = clamp(t_pixel_coord + make_float2(0, 2), make_float2(0), make_float2(t_res - 1));
		float4 b = t_tex.textureReadBilinear(b_pix, true);
		//top right
		float2 c_pix = clamp(t_pixel_coord + make_float2(2, 2), make_float2(0), make_float2(t_res - 1));
		float4 c = t_tex.textureReadBilinear(c_pix, true);
		//left
		float2 d_pix = clamp(t_pixel_coord + make_float2(-2, 0), make_float2(0), make_float2(t_res - 1));
		float4 d = t_tex.textureReadBilinear(d_pix, true);
		//center
		float4 e = t_tex.textureReadBilinear(t_pixel_coord, true);
		//right
		float2 f_pix = clamp(t_pixel_coord + make_float2(2, 0), make_float2(0), make_float2(t_res - 1));
		float4 f = t_tex.textureReadBilinear(f_pix, true);
		//bottom left
		float2 g_pix = clamp(t_pixel_coord + make_float2(-2, -2), make_float2(0), make_float2(t_res - 1));
		float4 g = t_tex.textureReadBilinear(g_pix, true);
		//bottom
		float2 h_pix = clamp(t_pixel_coord + make_float2(0, -2), make_float2(0), make_float2(t_res - 1));
		float4 h = t_tex.textureReadBilinear(h_pix, true);
		//bottom right
		float2 i_pix = clamp(t_pixel_coord + make_float2(2, -2), make_float2(0), make_float2(t_res - 1));
		float4 i = t_tex.textureReadBilinear(i_pix, true);

		//h 4x4 box
		float4 hb1 = lerp(lerp(a, b, 0.5), lerp(d, e, 0.5), 0.5);
		float4 hb2 = lerp(lerp(b, c, 0.5), lerp(e, f, 0.5), 0.5);
		float4 hb3 = lerp(lerp(d, e, 0.5), lerp(g, h, 0.5), 0.5);
		float4 hb4 = lerp(lerp(e, f, 0.5), lerp(h, i, 0.5), 0.5);

		//----
		//top left
		float2 j_pix = clamp(t_pixel_coord + make_float2(-1, 1), make_float2(0), make_float2(t_res - 1));
		float4 j = t_tex.textureReadBilinear((j_pix), true);
		//top
		float2 k_pix = clamp(t_pixel_coord + make_float2(1, 1), make_float2(0), make_float2(t_res - 1));
		float4 k = t_tex.textureReadBilinear((k_pix), true);
		//top right
		float2 l_pix = clamp(t_pixel_coord + make_float2(-1, -1), make_float2(0), make_float2(t_res - 1));
		float4 l = t_tex.textureReadBilinear((l_pix), true);
		//left
		float2 m_pix = clamp(t_pixel_coord + make_float2(1, -1), make_float2(0), make_float2(t_res - 1));
		float4 m = t_tex.textureReadBilinear((m_pix), true);

		//center h 4x4 box
		float4 hb5 = lerp(lerp(j, k, 0.5), lerp(l, m, 0.5), 0.5);

		float4 out = lerp(hb5, lerp(lerp(hb1, hb2, 0.5), lerp(hb3, hb4, 0.5), 0.5), 0.5);

		return out;
	}
	__device__ float4 texRead36TexelUV(DeviceTextureBuffer t_tex, int2 t_res, float2 uv_coord)
	{
		return make_float4(1);
	}
}

__global__ void downSample(const KittlesPT::GlobalShaderData t_shader_data, KittlesPT::DeviceTextureBuffer t_src, KittlesPT::DeviceTextureBuffer t_dst)
{
	using namespace KittlesPT;

	ShadingJob shading_job = getShadingJob(t_dst.dimensions);

	if (shading_job.invalid) {
		return;
	}

	float2 dst_uv = make_float2(shading_job.pixel_coord) / t_dst.dimensions;

	float2 src_pixel_coord = dst_uv * t_src.dimensions;
	src_pixel_coord = clamp(src_pixel_coord, make_float2(0), make_float2(t_src.dimensions - 1));

	//min filter
	float4 min_filtered_color = texRead36Texel(t_src, t_src.dimensions, src_pixel_coord);//TODO:pass as float2

	min_filtered_color = make_float4(clampOutput(make_float3(min_filtered_color)), 1);

	t_dst.textureWrite(min_filtered_color, shading_job.pixel_coord);
}

__global__ void upSampleCombine(const KittlesPT::GlobalShaderData t_shader_data, KittlesPT::DeviceTextureBuffer t_src, KittlesPT::DeviceTextureBuffer t_dst)
{
	using namespace KittlesPT;

	ShadingJob shading_job = getShadingJob(t_dst.dimensions);

	if (shading_job.invalid) {
		return;
	}

	float2 dst_uv = make_float2(shading_job.pixel_coord) / t_dst.dimensions;

	float2 src_pixel_coord = dst_uv * t_src.dimensions;
	src_pixel_coord = clamp(src_pixel_coord, make_float2(0.0f), make_float2(t_src.dimensions - 1));

	float4 mag_filtered_color = make_float4(0.0f);

	//Mag filter
	constexpr float gaussian_filter[3][3] = {
		{1 / 16.0f, 2 / 16.0f, 1 / 16.0f},
		{2 / 16.0f, 4 / 16.0f, 2 / 16.0f},
		{1 / 16.0f, 2 / 16.0f, 1 / 16.0f}
	};

	//TODO: make separable gaussian blur?
	for (int y = -1; y <= 1; y++)
	{
		for (int x = -1; x <= 1; x++)
		{
			float2 tap_pix = src_pixel_coord + make_float2(x, y);
			tap_pix = clamp(tap_pix, make_float2(0), make_float2(t_src.dimensions - 1));

			float4 tap_col = t_src.textureReadBilinear(tap_pix, true);

			mag_filtered_color += gaussian_filter[y + 1][x + 1] * tap_col;
		}
	}

	//combine prev mip
	float4 dst_mip_color = t_dst.textureReadNearest(make_float2(shading_job.pixel_coord));
	mag_filtered_color = lerp(dst_mip_color, mag_filtered_color, 0.75);

	mag_filtered_color = make_float4(clampOutput(make_float3(mag_filtered_color)), 1);

	t_dst.textureWrite(mag_filtered_color, shading_job.pixel_coord);
}