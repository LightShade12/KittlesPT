#pragma once

namespace KittlesPT
{
	struct GlobalShaderData;
	struct DeviceTextureBuffer;

	void launchPathTraceComputeKernel(const GlobalShaderData& shader_data);
	void launchPostProcessComputeKernel(const GlobalShaderData& shader_data);
	void launchBloomDownSampleComputeKernel(const GlobalShaderData& shader_data, const DeviceTextureBuffer& src, const DeviceTextureBuffer& dst, bool karis_avg);
	void launchBloomUpSampleComputeKernel(const GlobalShaderData& shader_data, const DeviceTextureBuffer& src, const DeviceTextureBuffer& dst);
}/*KittlesPT*/