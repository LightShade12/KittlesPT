#include "bloom.cuh"
#include "shading_kernel.cuh"

namespace KittlesPT
{
	__device__ float4 textureRead36Texels(const DeviceTextureBuffer& t_texture, float2 t_pixel_coord, bool karis_avg)
	{
		//Bilinear taps-------------------------
		//top left
		float2 a_pix = t_pixel_coord + make_float2(-2, 2);
		float4 a = t_texture.textureReadBilinear(a_pix, true);
		//top
		float2 b_pix = t_pixel_coord + make_float2(0, 2);
		float4 b = t_texture.textureReadBilinear(b_pix, true);
		//top right
		float2 c_pix = t_pixel_coord + make_float2(2, 2);
		float4 c = t_texture.textureReadBilinear(c_pix, true);
		//left
		float2 d_pix = t_pixel_coord + make_float2(-2, 0);
		float4 d = t_texture.textureReadBilinear(d_pix, true);
		//center
		float4 e = t_texture.textureReadBilinear(t_pixel_coord, true);
		//right
		float2 f_pix = t_pixel_coord + make_float2(2, 0);
		float4 f = t_texture.textureReadBilinear(f_pix, true);
		//bottom left
		float2 g_pix = t_pixel_coord + make_float2(-2, -2);
		float4 g = t_texture.textureReadBilinear(g_pix, true);
		//bottom
		float2 h_pix = t_pixel_coord + make_float2(0, -2);
		float4 h = t_texture.textureReadBilinear(h_pix, true);
		//bottom right
		float2 i_pix = t_pixel_coord + make_float2(2, -2);
		float4 i = t_texture.textureReadBilinear(i_pix, true);

		// H 4x4 box; color-coded in paper---------------
		float4 hb1 = (karis_avg) ? karisAverage(a, b, d, e) : lerp(lerp(a, b, 0.5), lerp(d, e, 0.5), 0.5);
		float4 hb2 = (karis_avg) ? karisAverage(b, c, e, f) : lerp(lerp(b, c, 0.5), lerp(e, f, 0.5), 0.5);
		float4 hb3 = (karis_avg) ? karisAverage(d, e, g, h) : lerp(lerp(d, e, 0.5), lerp(g, h, 0.5), 0.5);
		float4 hb4 = (karis_avg) ? karisAverage(e, f, h, i) : lerp(lerp(e, f, 0.5), lerp(h, i, 0.5), 0.5);

		//Central major 4x4 taps-----------------------------------------
		//top left
		float2 j_pix = t_pixel_coord + make_float2(-1, 1);
		float4 j = t_texture.textureReadBilinear(j_pix, true);
		//top
		float2 k_pix = t_pixel_coord + make_float2(1, 1);
		float4 k = t_texture.textureReadBilinear(k_pix, true);
		//top right
		float2 l_pix = t_pixel_coord + make_float2(-1, -1);
		float4 l = t_texture.textureReadBilinear(l_pix, true);
		//left
		float2 m_pix = t_pixel_coord + make_float2(1, -1);
		float4 m = t_texture.textureReadBilinear(m_pix, true);

		//Center H 4x4 box(Major weight)-----------------------------------------------
		float4 hb5 = (karis_avg) ? karisAverage(j, k, l, m) : lerp(lerp(j, k, 0.5), lerp(l, m, 0.5), 0.5);

		//Combination weights---------------------------------------------------
		float4 filtered_color = lerp(hb5, lerp(lerp(hb1, hb2, 0.5), lerp(hb3, hb4, 0.5), 0.5), 0.5);

		return filtered_color;
	}

	__device__ float4 textureRead36TexelsUV(const DeviceTextureBuffer& t_texture, float2 uv_coord, bool karis_avg)
	{
		float2 pixel_coord = uv_coord * t_texture.dimensions;
		return textureRead36Texels(t_texture, pixel_coord, karis_avg);
	}

	__device__ float4 karisAverage(float4 sp0, float4 sp1, float4 sp2, float4 sp3)
	{
		float w0 = 1.0f / RGBSpectrum(sp0).getLuminance();
		float w1 = 1.0f / RGBSpectrum(sp1).getLuminance();
		float w2 = 1.0f / RGBSpectrum(sp2).getLuminance();
		float w3 = 1.0f / RGBSpectrum(sp3).getLuminance();

		float net_w = 1.0f / (w0 + w1 + w2 + w3);

		return (sp0 * w0 + sp1 * w1 + sp2 * w2 + sp3 * w3) * net_w;
	}

	__constant__ constexpr float GAUSSIAN_3x3_KERNEL[3][3] = {
		{1.0f / 16.0f, 2.0f / 16.0f, 1.0f / 16.0f},
		{2.0f / 16.0f, 4.0f / 16.0f, 2.0f / 16.0f},
		{1.0f / 16.0f, 2.0f / 16.0f, 1.0f / 16.0f}
	};
}/*KittlesPT*/

__global__ void downSample(const KittlesPT::GlobalShaderData t_shader_data, KittlesPT::DeviceTextureBuffer t_src, KittlesPT::DeviceTextureBuffer t_dst, bool karis_avg)
{
	using namespace KittlesPT;

	ShadingJob shading_job = getShadingJob(t_dst.dimensions);

	if (shading_job.invalid) {
		return;
	}

	//TODO: use shading_job.uv_coord?
	float2 dst_uv = make_float2(shading_job.pixel_coord) / t_dst.dimensions;
	float2 src_pixel_coord = dst_uv * t_src.dimensions;

	//Min filter
	float4 min_filtered_color = textureRead36Texels(t_src, src_pixel_coord, karis_avg);//TODO:pass as float2
	min_filtered_color = make_float4(clampOutput(make_float3(min_filtered_color)), 1);

	t_dst.textureWrite(min_filtered_color, shading_job.pixel_coord);
}

__global__ void upSampleCombine(const KittlesPT::GlobalShaderData shader_data, KittlesPT::DeviceTextureBuffer t_src,
	KittlesPT::DeviceTextureBuffer t_dst)
{
	using namespace KittlesPT;

	ShadingJob shading_job = getShadingJob(t_dst.dimensions);

	if (shading_job.invalid) {
		return;
	}

	float2 dst_uv = make_float2(shading_job.pixel_coord) / t_dst.dimensions;
	float2 src_pixel_coord = dst_uv * t_src.dimensions;

	float4 mag_filtered_color{ 0.0f,0.0f,0.0f,0.0f };

	//TODO: make separable gaussian blur?
	//Mag filter
#pragma unroll
	for (int y = -1; y <= 1; y++) {
#pragma unroll
		for (int x = -1; x <= 1; x++) {
			float2 tap_pixel_coord = src_pixel_coord + make_float2(x, y);
			float4 tap_color = t_src.textureReadBilinear(tap_pixel_coord, true);
			mag_filtered_color += GAUSSIAN_3x3_KERNEL[y + 1][x + 1] * tap_color;
		}
	}

	//combine prev mip
	float4 dst_prev_color = t_dst.textureReadNearest(make_float2(shading_job.pixel_coord));
	mag_filtered_color = lerp(dst_prev_color, mag_filtered_color, shader_data.renderer_settings.bloom_internal_blend);
	mag_filtered_color = make_float4(clampOutput(make_float3(mag_filtered_color)), 1.0f);

	t_dst.textureWrite(mag_filtered_color, shading_job.pixel_coord);
}