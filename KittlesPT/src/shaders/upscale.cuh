#pragma once
#include "shading_kernel.cuh"
#include "containers.cuh"

namespace KittlesPT
{
	__global__ void upscale(KittlesPT::DeviceTextureBuffer t_src, KittlesPT::DeviceTextureBuffer t_dst)
	{
		ShadingJob shading_job = getShadingJob(t_dst.dimensions);

		if (shading_job.is_invalid) {
			return;
		}

		float2 dst_uv = make_float2(shading_job.pixel_coord) / t_dst.dimensions;
		float2 src_pixel_coord = dst_uv * t_src.dimensions;

		float4 data = t_src.textureReadBilinear(src_pixel_coord, true);
		t_dst.textureWrite(data, shading_job.pixel_coord);
	}
}