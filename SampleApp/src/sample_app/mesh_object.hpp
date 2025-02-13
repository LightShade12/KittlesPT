#pragma once

#include "kittlesPT/kittlesPT.hpp"
#include "glm/gtc/matrix_transform.hpp"
#include "glm/gtc/quaternion.hpp"
#include "glm/glm.hpp"

namespace SampleApp
{
	class MeshObject
	{
	public:
		MeshObject() = default;

		MeshObject(int32_t mesh_id, const glm::mat4& original_model_matrix) :
			mesh_index(mesh_id), original_transform(original_model_matrix)
		{}

		void updateMeshTransform(KittlesPT::Renderer& renderer)
		{
			glm::mat4 translation_matrix = glm::translate(glm::mat4(1), translation);
			glm::mat4 scale_matrix = glm::scale(glm::mat4(1.0f), scale);

			/*glm::mat4 rot_x = glm::rotate(glm::mat4(1), glm::radians(rotation.x), glm::vec3(1, 0, 0));
			glm::mat4 rot_y = glm::rotate(glm::mat4(1), glm::radians(rotation.y), glm::vec3(0, 1, 0));
			glm::mat4 rot_z = glm::rotate(glm::mat4(1), glm::radians(rotation.z), glm::vec3(0, 0, 1));*/
			glm::mat4 rotation_matrix = glm::mat4_cast(glm::quat(glm::radians(rotation)));

			glm::mat4 model_matrix = original_transform * translation_matrix * rotation_matrix * scale_matrix;
			renderer.setMeshTransform(mesh_index, model_matrix);
		}

		int32_t mesh_index = -1;
		glm::vec3 translation = glm::vec3(0.0f);
		glm::vec3 scale = glm::vec3(1.0f);
		glm::vec3 rotation = glm::vec3(0.0f);//TODO: Euler? Quat?
		//glm::quat rotation = glm::quat(glm::radians(glm::vec3(0.0f)));

		glm::mat4 original_transform;
	};
}