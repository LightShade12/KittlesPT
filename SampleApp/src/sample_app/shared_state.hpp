#pragma once

namespace SampleApp
{
	struct RendererSettings
	{
		float exposure = 0;
		float fov_y_rad = 0;
	};

	struct ApplicationData
	{
		RendererSettings renderer;
	};
}