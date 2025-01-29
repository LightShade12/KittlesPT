#pragma once

#include "renderer.hpp"
#include "glm/glm.hpp"

namespace SampleApp
{
	struct ApplicationData
	{
		KittlesPT::ProceduralEnvironmentData environment_data;
		KittlesPT::RendererSettings renderer_settings;
		KittlesPT::MaterialSceneEntity editable_material;
		int editable_material_idx = 0;
		int editable_mesh_idx = 0;
		size_t materials_count = 0;
		size_t meshes_count = 0;
		glm::mat4 mesh_transform = glm::mat4(1.0f);
	};
}