#pragma once
#include "kittlesPT/kittlesPT.hpp"

#include "tinygltf/tiny_gltf.h"
#include "glm/glm.hpp"
#define GLM_ENABLE_EXPERIMENTAL
#include "glm/gtx/quaternion.hpp"

#include "stb/stb_image.h"

#include <iostream>
#include <chrono>

namespace SampleApp
{
	class ModelImporter
	{
	public:
		ModelImporter() = default;

		bool loadGLTFfromFile(const char* file_path, KittlesPT::BasicScene* scene);

	private:
		KittlesPT::BasicScene* m_scene = nullptr;
		tinygltf::Model m_scene_model;
		bool m_is_binary_file = false;

		bool loadModel(const char* file_path);
		bool loadTextures();
		bool loadMaterials();

		bool parseMesh(const tinygltf::Node& mesh_node);
		bool parseCamera(const tinygltf::Node& camera_node);
		bool parseNode(const tinygltf::Node& node);

		bool extractVerticeAttributes(
			const tinygltf::Mesh& mesh,
			std::vector<glm::vec3>* positions,
			std::vector<glm::vec3>* normals,
			std::vector<glm::vec2>* tex_coords,
			std::vector<int>* primitive_mat_idx);
	};

	static std::string getFilePathExtension(const std::string& file_path) {
		if (file_path.find_last_of(".") != std::string::npos)
			return file_path.substr(file_path.find_last_of(".") + 1);
		return "";
	}

	//TODO:more coordinated loading
	bool ModelImporter::loadGLTFfromFile(const char* file_path, KittlesPT::BasicScene* scene)
	{
		std::cerr << "[Importer] -----------------Starting loading-------------------\n";
		std::chrono::time_point<std::chrono::steady_clock> start = std::chrono::high_resolution_clock::now();

		m_scene = scene;
		if (!loadModel(file_path)) {
			std::cerr << "[Importer] Error occured while loading GLTF model\n";
			return false;
		}

		std::cerr << "[Importer] Extensions report:\n";
		for (std::string extensionname : m_scene_model.extensionsUsed) {
			printf("> using: %s\n", extensionname.c_str());
		}
		for (std::string extensionname : m_scene_model.extensionsRequired) {
			printf("> required: %s\n", extensionname.c_str());
		}
		std::cerr << "[Importer] End of extensions report\n";

		if (!loadTextures()) {
			std::cerr << "[Importer] Image loading stage failure\n";
		};
		if (!loadMaterials()) {
			std::cerr << "[Importer] Material loading stage failure\n";
		};

		printf("\n[Importer] Detected nodes in file:%zu\n", m_scene_model.nodes.size());
		printf("[Importer] Detected meshes in file:%zu\n", m_scene_model.meshes.size());
		printf("[Importer] Detected cameras in file:%zu\n\n", m_scene_model.cameras.size());

		int scene_index = 0;
		//node looping
		for (size_t node_idx = 0; node_idx < m_scene_model.scenes[scene_index].nodes.size(); node_idx++)
		{
			const tinygltf::Node& gltf_node = m_scene_model.nodes[m_scene_model.scenes[scene_index].nodes[node_idx]];
			printf("[Importer] processing node: %s\n", gltf_node.name.c_str());

			if (gltf_node.children.size() > 0) {
				parseNode(gltf_node);
			}

			if (gltf_node.camera >= 0) {
				parseCamera(gltf_node);
			}

			if (gltf_node.mesh >= 0) {
				parseMesh(gltf_node);
			}
		}

		std::chrono::time_point<std::chrono::steady_clock> end = std::chrono::high_resolution_clock::now();
		std::chrono::duration<float> delay_secs = (end - start);
		std::cerr << "[Importer]-----------------Loading finished [took " << delay_secs.count() << " secs]-----------------\n";

		return true;
	}

	bool ModelImporter::loadModel(const char* file_path)
	{
		tinygltf::TinyGLTF loader;
		std::string err;
		std::string warn;
		bool load_success = false;

		std::string extension = getFilePathExtension(file_path);

		//Load
		if (extension.compare("glb") == 0) {
			load_success = loader.LoadBinaryFromFile(&m_scene_model, &err, &warn, file_path);
			m_is_binary_file = true;
		}
		else {
			load_success = loader.LoadASCIIFromFile(&m_scene_model, &err, &warn, file_path);
		}

		if (!warn.empty()) {
			std::cerr << "[Importer] WARN: " << warn << std::endl;
		}
		if (!err.empty()) {
			std::cerr << "[Importer] ERR: " << err << std::endl;
		}

		if (!load_success) {
			std::cerr << "[Importer] Failed to load glTF: " << file_path << std::endl;
		}
		else {
			std::cerr << "[Importer] Loaded glTF: " << file_path << std::endl;
		}

		return load_success;
	}

	bool ModelImporter::parseMesh(const tinygltf::Node& mesh_node)
	{
		tinygltf::Mesh gltf_mesh = m_scene_model.meshes[mesh_node.mesh];
		printf("> processing mesh:%s\n", gltf_mesh.name.c_str());

		std::vector<glm::vec3> loaded_mesh_positions;
		std::vector<glm::vec3>loaded_mesh_normals;
		std::vector<glm::vec2>loaded_mesh_tex_coods;
		std::vector<int>loaded_mesh_primitive_mat_id;

		extractVerticeAttributes(gltf_mesh,
			&loaded_mesh_positions, &loaded_mesh_normals,
			&loaded_mesh_tex_coods, &loaded_mesh_primitive_mat_id);

		glm::mat4 model_matrix = glm::identity<glm::mat4>();
		if (mesh_node.matrix.size() > 0) {
			model_matrix = glm::mat4(
				mesh_node.matrix[0], mesh_node.matrix[1], mesh_node.matrix[2], mesh_node.matrix[3],
				mesh_node.matrix[4], mesh_node.matrix[5], mesh_node.matrix[6], mesh_node.matrix[7],
				mesh_node.matrix[8], mesh_node.matrix[9], mesh_node.matrix[10], mesh_node.matrix[11],
				mesh_node.matrix[12], mesh_node.matrix[13], mesh_node.matrix[14], mesh_node.matrix[15]
			);
		}
		else {
			glm::mat4 scale_mat = glm::identity<glm::mat4>();
			glm::mat4 rot_mat = glm::identity<glm::mat4>();
			glm::mat4 translation_mat = glm::identity<glm::mat4>();

			if (mesh_node.scale.size() > 0) {
				scale_mat = glm::scale(glm::mat4(1.0f),
					glm::vec3(
						(float)mesh_node.scale[0],
						(float)mesh_node.scale[1],
						(float)mesh_node.scale[2]));
			}

			if (mesh_node.rotation.size() > 0) {
				glm::quat quaternion = glm::quat(
					(float)mesh_node.rotation[3],
					(float)mesh_node.rotation[0],
					(float)mesh_node.rotation[1],
					(float)mesh_node.rotation[2]);
				rot_mat = glm::toMat4(quaternion);
			}

			if (mesh_node.translation.size() > 0) {
				translation_mat = glm::translate(glm::mat4(1.0f),
					glm::vec3(
						(float)mesh_node.translation[0],
						(float)mesh_node.translation[1],
						(float)mesh_node.translation[2]));
			}
			//TRS
			model_matrix = translation_mat * rot_mat * scale_mat;
		}

		KittlesPT::MeshSceneEntity mesh(gltf_mesh.name, model_matrix);

		//Positions.size() and vertex_normals.size() must be equal!
		if (loaded_mesh_positions.size() != loaded_mesh_normals.size()) {
			printf("\n>> [POSITIONS-NORMALS COUNT MISMATCH] !\n");
			return false;
		}
		//contruct and push tris
		for (size_t i = 0; i < loaded_mesh_positions.size(); i += 3)
		{
			int mtidx = loaded_mesh_primitive_mat_id[i / 3];

			mesh.addShape(KittlesPT::TriangleSceneEntity(
				loaded_mesh_positions[i], loaded_mesh_positions[i + 1], loaded_mesh_positions[i + 2],
				loaded_mesh_normals[i], loaded_mesh_normals[i + 1], loaded_mesh_normals[i + 2],
				loaded_mesh_tex_coods[i], loaded_mesh_tex_coods[i + 1], loaded_mesh_tex_coods[i + 2],
				mtidx)
			);
		}
		m_scene->addMesh(mesh);

		return false;
	}

	bool ModelImporter::extractVerticeAttributes(
		const tinygltf::Mesh& mesh,
		std::vector<glm::vec3>* positions,
		std::vector<glm::vec3>* normals,
		std::vector<glm::vec2>* tex_coords,
		std::vector<int>* primitive_mat_idx)
	{
		for (size_t prim_idx = 0; prim_idx < mesh.primitives.size(); prim_idx++)
		{
			tinygltf::Primitive primitive = mesh.primitives[prim_idx];

			int pos_attrib_accesor_idx = primitive.attributes["POSITION"];
			int nrm_attrib_accesor_idx = primitive.attributes["NORMAL"];
			int uv_attrib_accesor_idx = primitive.attributes["TEXCOORD_0"];
			int indices_accessor_idx = primitive.indices;

			tinygltf::Accessor pos_accesor = m_scene_model.accessors[pos_attrib_accesor_idx];
			tinygltf::Accessor nrm_accesor = m_scene_model.accessors[nrm_attrib_accesor_idx];
			tinygltf::Accessor uv_accesor = m_scene_model.accessors[uv_attrib_accesor_idx];
			tinygltf::Accessor indices_accesor = m_scene_model.accessors[indices_accessor_idx];

			tinygltf::BufferView pos_bufferview = m_scene_model.bufferViews[pos_accesor.bufferView];
			tinygltf::BufferView nrm_bufferview = m_scene_model.bufferViews[nrm_accesor.bufferView];
			tinygltf::BufferView uv_bufferview = m_scene_model.bufferViews[uv_accesor.bufferView];
			tinygltf::BufferView indices_bufferview = m_scene_model.bufferViews[indices_accesor.bufferView];

			uint64_t pos_buffer_byte_offset = pos_bufferview.byteOffset;
			uint64_t nrm_buffer_byte_offset = nrm_bufferview.byteOffset;
			uint64_t uv_buffer_byte_offset = uv_bufferview.byteOffset;

			tinygltf::Buffer indices_buffer = m_scene_model.buffers[indices_bufferview.buffer];//should alawys be zero?

			unsigned short* indicesbuffer = (unsigned short*)(indices_buffer.data.data());
			glm::vec3* positions_buffer = (glm::vec3*)(indices_buffer.data.data() + pos_buffer_byte_offset);
			glm::vec3* normals_buffer = (glm::vec3*)(indices_buffer.data.data() + nrm_buffer_byte_offset);
			glm::vec2* UVs_buffer = (glm::vec2*)(indices_buffer.data.data() + uv_buffer_byte_offset);

			for (uint64_t i = (indices_bufferview.byteOffset / 2); i < (indices_bufferview.byteLength + indices_bufferview.byteOffset) / 2; i++)
			{
				positions->push_back(positions_buffer[indicesbuffer[i]]);
				normals->push_back(normals_buffer[indicesbuffer[i]]);
				tex_coords->push_back(UVs_buffer[indicesbuffer[i]]);
			}
			for (size_t i = 0; i < indices_accesor.count / 3; i++)//no of triangles per primitive
			{
				primitive_mat_idx->push_back(primitive.material);
			}
		}
		return true;
	}

	bool ModelImporter::loadTextures()
	{
		const char* image_reference_directory = "../models/";//TODO:bad
		printf("[Importer] detected textures count in file: %zu\n", m_scene_model.images.size());

		for (size_t texture_idx = 0; texture_idx < m_scene_model.images.size(); texture_idx++)
		{
			tinygltf::Image gltf_image = m_scene_model.images[texture_idx];
			printf("[Importer] loading image: %s\n", gltf_image.name.c_str());

			int width = 0, height = 0, numcolch = 0;
			const unsigned char* finalimgdata = nullptr;

			if (m_is_binary_file)
			{
				tinygltf::BufferView imgbufferview = m_scene_model.bufferViews[gltf_image.bufferView];
				const unsigned char* imgdata = m_scene_model.buffers[imgbufferview.buffer].data.data() + imgbufferview.byteOffset;
				size_t byte_len = imgbufferview.byteLength;

				finalimgdata = stbi_load_from_memory(imgdata, (int)byte_len,
					&width, &height, &numcolch, 3);
			}
			else
			{
				finalimgdata = stbi_load((image_reference_directory + gltf_image.uri).c_str(),
					&width, &height, &numcolch, 3);
			}
			if (finalimgdata == nullptr) {
				printf("[Importer] Image data was null!\n");
				return false;
			}

			printf("> name: %s | dims: %d x %d | channels: %d------\n", gltf_image.name.c_str(), width, height, numcolch);
			//TODO: avoid casting away constness
			m_scene->addTexture(KittlesPT::TextureSceneEntity(const_cast<unsigned char*>(finalimgdata), width, height, 3));

			stbi_image_free((void*)finalimgdata);
		}
		return true;
	}

	bool ModelImporter::loadMaterials()
	{
		printf("[Importer] detected materials count in file: %zu\n", m_scene_model.materials.size());

		for (size_t mat_idx = 0; mat_idx < m_scene_model.materials.size(); mat_idx++)
		{
			tinygltf::Material gltf_material = m_scene_model.materials[mat_idx];
			printf("[Importer] loading material: %s\n", gltf_material.name.c_str());
			tinygltf::PbrMetallicRoughness PBR_data = gltf_material.pbrMetallicRoughness;
			int albedo_tex_id = -1;
			int ORM_tex_id = -1;
			int transmission_tex_id = -1;
			int emission_tex_id = -1;
			int normal_tex_id = -1;
			float transmission = 0.0f;
			float ior = 1.45f;
			float emission_scale_nits = 0.0f;

			//TODO: We just use RGB material albedo for now
			glm::vec3 albedo_factor = glm::vec3(PBR_data.baseColorFactor[0], PBR_data.baseColorFactor[1], PBR_data.baseColorFactor[2]);
			if (PBR_data.baseColorTexture.index >= 0) {
				albedo_tex_id = m_scene_model.textures[PBR_data.baseColorTexture.index].source;
			}

			if (PBR_data.metallicRoughnessTexture.index >= 0) {
				ORM_tex_id = m_scene_model.textures[PBR_data.metallicRoughnessTexture.index].source;
			}

			if (gltf_material.normalTexture.index >= 0) {
				normal_tex_id = m_scene_model.textures[gltf_material.normalTexture.index].source;
			}

			glm::vec3 emission_factor = glm::vec3(gltf_material.emissiveFactor[0], gltf_material.emissiveFactor[1], gltf_material.emissiveFactor[2]);
			if (gltf_material.emissiveTexture.index >= 0) {
				emission_tex_id = m_scene_model.textures[gltf_material.emissiveTexture.index].source;
			}

			if (gltf_material.extensions.find("KHR_materials_transmission") != gltf_material.extensions.end()) {
				transmission = (float)gltf_material.extensions["KHR_materials_transmission"].Get("transmissionFactor").GetNumberAsDouble();
				int tex_id = gltf_material.extensions["KHR_materials_transmission"].Get("transmissionTexture").Get("index").GetNumberAsInt();
				if (tex_id >= 0) {
					transmission_tex_id = m_scene_model.textures[tex_id].source;
				}
			};
			if (gltf_material.extensions.find("KHR_materials_ior") != gltf_material.extensions.end()) {
				ior = (float)gltf_material.extensions["KHR_materials_ior"].Get("ior").GetNumberAsDouble();
			};
			if (gltf_material.extensions.find("KHR_materials_emissive_strength") != gltf_material.extensions.end()) {
				emission_scale_nits = (float)gltf_material.extensions["KHR_materials_emissive_strength"].Get("emissiveStrength").GetNumberAsDouble();
			};

			m_scene->addMaterial(KittlesPT::MaterialSceneEntity(
				albedo_tex_id, albedo_factor,
				ORM_tex_id, (float)PBR_data.metallicFactor, (float)PBR_data.roughnessFactor,
				transmission_tex_id, transmission, ior,
				emission_tex_id, emission_factor, emission_scale_nits,
				normal_tex_id, (float)gltf_material.normalTexture.scale
			));
		}

		return true;
	}

	bool ModelImporter::parseCamera(const tinygltf::Node& camera_node)
	{
		tinygltf::Camera gltf_camera = m_scene_model.cameras[camera_node.camera];
		printf("\n> found a camera: %s\n", gltf_camera.name.c_str());

		glm::mat4 model_mat(1.0f);
		if (camera_node.matrix.empty())
		{
			printf("> found decomposed matrices\n");
			glm::mat4 translation_mat = glm::identity<glm::mat4>();
			glm::mat4 rot_mat = glm::identity<glm::mat4>();

			if (camera_node.rotation.size() > 0) {
				printf("> found rotation matrix\n");

				glm::quat quaternion = glm::quat(
					(float)camera_node.rotation[3],
					(float)camera_node.rotation[0],
					(float)camera_node.rotation[1],
					(float)camera_node.rotation[2]);
				rot_mat = glm::toMat4(glm::normalize(quaternion));
			}

			if (camera_node.translation.size() > 0)
			{
				printf("> found translation matrix\n");

				translation_mat = glm::translate(glm::mat4(1.0f),
					glm::vec3(
						(float)camera_node.translation[0],
						(float)camera_node.translation[1],
						(float)camera_node.translation[2]));
			}
			model_mat = translation_mat * rot_mat;
		}
		else {
			auto& m = camera_node.matrix;
			printf("> found a model matrix\n");
			model_mat = glm::mat4(
				m[0], m[1], m[2], m[3],
				m[4], m[5], m[6], m[7],
				m[8], m[9], m[10], m[11],
				m[12], m[13], m[14], m[15]);
		}
		glm::mat4 view_matrix = glm::inverse(model_mat);
		printf("> x:%.3f y:%.3f z:%.3f\n", camera_node.translation[0], camera_node.translation[1], camera_node.translation[2]);
		printf("> x:%.3f y:%.3f z:%.3f\n", model_mat[3][0], model_mat[3][1], model_mat[3][2]);
		printf("> x:%.3f y:%.3f z:%.3f\n", view_matrix[3][0], view_matrix[3][1], view_matrix[3][2]);
		KittlesPT::CameraSceneEntity camera(gltf_camera.name, view_matrix, (float)gltf_camera.perspective.yfov);
		m_scene->addCamera(camera);

		return true;
	}

	bool ModelImporter::parseNode(const tinygltf::Node& node)
	{
		return true;
		//node looping
		for (size_t nodeIdx = 0; nodeIdx < node.children.size(); nodeIdx++)
		{
			tinygltf::Node gltf_node = m_scene_model.nodes[nodeIdx];
			printf("[Importer] processing node: %s\n", gltf_node.name.c_str());

			if (gltf_node.children.size() > 0) {
				parseNode(gltf_node);
			}

			if (gltf_node.camera >= 0) {
				parseCamera(gltf_node);
			}

			if (gltf_node.mesh >= 0) {
				parseMesh(gltf_node);
			}
		}
		return false;
	}
}