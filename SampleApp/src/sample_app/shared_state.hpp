#pragma once

#include "renderer.hpp"
#include "glm/glm.hpp"

namespace SampleApp
{
	//TODO: replace with material scene entity
	struct Material
	{
		glm::vec3 albedo;
		float metallicity = 0.0f;
		float roughness = 0.5f;
		float transmission = 0.0f;
		float ior = 1.45f;
	};

	struct ApplicationData
	{
		KittlesPT::ProceduralEnvironmentData environment_data;
		KittlesPT::PathtracerSettings pathtracer_settings;
		Material editable_material;
		int editable_material_idx = 0;
		int materials_count = 0;
	};
}