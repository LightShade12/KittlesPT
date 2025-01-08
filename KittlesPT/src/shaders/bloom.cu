#include "bloom.cuh"

namespace KittlesPT
{
	__device__ float4 texReadCODAW(DeviceTextureBuffer t_tex, int2 t_res, int2 t_pixel_coord)
	{
		//top left
		int2 a_pix = clamp(t_pixel_coord + make_int2(-2, 2), make_int2(0), t_res - 1);
		float4 a = t_tex.textureReadBilinear(make_float2(a_pix) + 0.5f, true);
		//top
		int2 b_pix = clamp(t_pixel_coord + make_int2(0, 2), make_int2(0), t_res - 1);
		float4 b = t_tex.textureReadBilinear(make_float2(b_pix) + 0.5f, true);
		//top right
		int2 c_pix = clamp(t_pixel_coord + make_int2(2, 2), make_int2(0), t_res - 1);
		float4 c = t_tex.textureReadBilinear(make_float2(c_pix) + 0.5f, true);
		//left
		int2 d_pix = clamp(t_pixel_coord + make_int2(-2, 0), make_int2(0), t_res - 1);
		float4 d = t_tex.textureReadBilinear(make_float2(d_pix) + 0.5f, true);
		//center
		float4 e = t_tex.textureReadBilinear(make_float2(t_pixel_coord) + 0.5f, true);
		//right
		int2 f_pix = clamp(t_pixel_coord + make_int2(2, 0), make_int2(0), t_res - 1);
		float4 f = t_tex.textureReadBilinear(make_float2(f_pix) + 0.5f, true);
		//bottom left
		int2 g_pix = clamp(t_pixel_coord + make_int2(-2, -2), make_int2(0), t_res - 1);
		float4 g = t_tex.textureReadBilinear(make_float2(g_pix) + 0.5f, true);
		//bottom
		int2 h_pix = clamp(t_pixel_coord + make_int2(0, -2), make_int2(0), t_res - 1);
		float4 h = t_tex.textureReadBilinear(make_float2(h_pix) + 0.5f, true);
		//bottom right
		int2 i_pix = clamp(t_pixel_coord + make_int2(2, -2), make_int2(0), t_res - 1);
		float4 i = t_tex.textureReadBilinear(make_float2(i_pix) + 0.5f, true);

		//h 4x4 box
		float4 hb1 = lerp(lerp(a, b, 0.5), lerp(d, e, 0.5), 0.5);
		float4 hb2 = lerp(lerp(b, c, 0.5), lerp(e, f, 0.5), 0.5);
		float4 hb3 = lerp(lerp(d, e, 0.5), lerp(g, h, 0.5), 0.5);
		float4 hb4 = lerp(lerp(e, f, 0.5), lerp(h, i, 0.5), 0.5);

		//----
		//top left
		int2 j_pix = clamp(t_pixel_coord + make_int2(-1, 1), make_int2(0), t_res - 1);
		float4 j = t_tex.textureReadBilinear(make_float2(j_pix) + 0.5f, true);
		//top
		int2 k_pix = clamp(t_pixel_coord + make_int2(1, 1), make_int2(0), t_res - 1);
		float4 k = t_tex.textureReadBilinear(make_float2(k_pix) + 0.5f, true);
		//top right
		int2 l_pix = clamp(t_pixel_coord + make_int2(-1, -1), make_int2(0), t_res - 1);
		float4 l = t_tex.textureReadBilinear(make_float2(l_pix) + 0.5f, true);
		//left
		int2 m_pix = clamp(t_pixel_coord + make_int2(1, -1), make_int2(0), t_res - 1);
		float4 m = t_tex.textureReadBilinear(make_float2(m_pix) + 0.5f, true);

		//center h 4x4 box
		float4 hb5 = lerp(lerp(j, k, 0.5), lerp(l, m, 0.5), 0.5);

		float4 out = lerp(hb5, lerp(lerp(hb1, hb2, 0.5), lerp(hb3, hb4, 0.5), 0.5), 0.5);

		return out;
	}
}

__global__ void downSample(const KittlesPT::GlobalShaderData t_shader_data, KittlesPT::DeviceTextureBuffer t_src, KittlesPT::DeviceTextureBuffer t_dst)
{
	using namespace KittlesPT;
	//setup threads
	int thread_pixel_coord_x = threadIdx.x + blockIdx.x * blockDim.x;
	int thread_pixel_coord_y = threadIdx.y + blockIdx.y * blockDim.y;
	int2 pixel_coord = make_int2(thread_pixel_coord_x, thread_pixel_coord_y);

	//int2 frame_res = t_shader_data.frame_resolution;
	//float2 uv_coord = { (float)pixel_coord.x / (float)frame_res.x, (float)pixel_coord.y / (float)frame_res.y };

	if ((pixel_coord.x >= t_dst.width) || (pixel_coord.y >= t_dst.height)) {
		return;
	}
	//=========================================================

	float2 scale_ratio = make_float2((float)t_src.width / t_dst.width, (float)t_src.height / t_dst.height);

	//src-res
	int2 src_tap_pix = make_int2(pixel_coord.x * scale_ratio.x, pixel_coord.y * scale_ratio.y);
	src_tap_pix = clamp(src_tap_pix, make_int2(0), make_int2(t_src.width, t_src.height) - 1);

	//min filter
	//float4 color = texReadBilinear(t_src, make_float2(src_tap_pix) + 0.5f, t_src_res, false);
	float4 color = texReadCODAW(t_src, make_int2(t_src.width, t_src.height), src_tap_pix);

	t_dst.textureWrite(color, pixel_coord);
}

__global__ void upSampleCombine(const KittlesPT::GlobalShaderData t_shader_data, KittlesPT::DeviceTextureBuffer t_src, KittlesPT::DeviceTextureBuffer t_dst, bool combine)
{
	using namespace KittlesPT;

	//setup threads
	int thread_pixel_coord_x = threadIdx.x + blockIdx.x * blockDim.x;
	int thread_pixel_coord_y = threadIdx.y + blockIdx.y * blockDim.y;
	int2 pixel_coord = make_int2(thread_pixel_coord_x, thread_pixel_coord_y);

	//int2 frame_res = t_shader_data.frame_resolution;
	//float2 uv_coord = { (float)pixel_coord.x / (float)frame_res.x, (float)pixel_coord.y / (float)frame_res.y };

	if ((pixel_coord.x >= t_dst.width) || (pixel_coord.y >= t_dst.height)) {
		return;
	}
	//=========================================================

	float2 scale_down_ratio = make_float2((float)pixel_coord.x / (float)t_dst.width, (float)pixel_coord.y / (float)t_dst.height);

	float2 src_pixf = make_float2(scale_down_ratio.x * t_src.width, scale_down_ratio.y * t_src.height);

	src_pixf = clamp(src_pixf, make_float2(0.0f), make_float2(t_src.width, t_src.height) - 1.0f);

	float4 final_col = make_float4(0.0f);
	//float4 final_col = texReadBilinear(t_src, src_pixf, t_src_res, true);

	constexpr float filter[3][3] = {
		{1 / 16.0f, 2 / 16.0f, 1 / 16.0f},
		{2 / 16.0f, 4 / 16.0f, 2 / 16.0f},
		{1 / 16.0f, 2 / 16.0f, 1 / 16.0f}
	};

	//TODO: make separable gaussian blur?
	for (int y = -1; y <= 1; y++)
	{
		for (int x = -1; x <= 1; x++)
		{
			float2 tap_pix = src_pixf + make_float2(x, y);
			tap_pix = clamp(tap_pix, make_float2(0), make_float2(t_src.width, t_src.height) - 1.0f);

			float4 tap_col = t_src.textureReadBilinear(tap_pix, true);

			final_col += filter[y + 1][x + 1] * tap_col;
		}
	}

	if (combine)
	{
		//combine prev mip
		float4 col = t_dst.textureReadNearest(pixel_coord);
		//final_col += col;
		final_col = lerp(col, final_col, 0.75);
	}

	t_dst.textureWrite(final_col, pixel_coord);
}