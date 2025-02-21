#pragma once

#define USE_FSR

namespace KittlesPT
{
	struct ShaderData;
	struct DeviceTextureBuffer;

	void launchPathTraceComputeMegaKernel(const ShaderData& shader_data);
	void launchUpscaleComputeKernel(const DeviceTextureBuffer& src, const DeviceTextureBuffer& dst, const DeviceTextureBuffer& back);
	void launchPostProcessComputeKernel(const ShaderData& shader_data);
	void launchFxComputeKernel(const ShaderData& shader_data);
	void launchModulateComputeKernel(const ShaderData& shader_data);
	void launchHistogramComputeKernel(const ShaderData& shader_data);
	void launchHistogramAverageComputeKernel(const ShaderData& shader_data);
	void launchBloomDownSampleComputeKernel(const ShaderData& shader_data, const DeviceTextureBuffer& src, const DeviceTextureBuffer& dst,
		bool karis_avg);
	void launchBloomUpSampleComputeKernel(const ShaderData& shader_data, const DeviceTextureBuffer& src, const DeviceTextureBuffer& dst);
}/*KittlesPT*/