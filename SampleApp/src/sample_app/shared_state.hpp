#pragma once

#include "renderer.hpp"
#include "mesh_object.hpp"
#include "glm/glm.hpp"

namespace SampleApp
{
	struct ApplicationData
	{
		KittlesPT::ProceduralEnvironmentSettings environment_settings;
		KittlesPT::RendererSettings renderer_settings;
		//----------
		KittlesPT::MaterialSceneEntity editable_material;
		int32_t editable_material_idx = 0;
		size_t materials_count = 0;

		MeshObject editable_mesh_object;
		int32_t editable_mesh_idx = 0;
		size_t meshes_count = 0;
	};
}