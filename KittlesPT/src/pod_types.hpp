#pragma once

namespace KittlesPT
{
	struct ProceduralEnvironmentData
	{
		float sun_angular_diameter_rad = 0.0087f;
		float sun_phi_rad = 4.18879;
		float sun_theta_rad = 0.785f;
		float sun_radiance_intensity = 0.0f;//def: 50.0f
	};

	struct PathtracerSettings
	{
		int max_bounce_depth = 3;
	};
}