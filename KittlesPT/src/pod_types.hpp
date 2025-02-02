#pragma once
#include "glm/glm.hpp"
#include "glm/gtc/constants.hpp"
#include <vector>
#include <string>

namespace KittlesPT
{
	struct ProceduralEnvironmentData
	{
		//sun: 31831 nits; 100000 lx
		//sky: 1910 nits; 6000 lx
		float sun_angular_diameter_rad = 0.0087f;
		float sun_phi_rad = 4.18879f;
		//float sun_theta_rad = 0.785f;
		float sun_theta_rad = 0.0872665f;
		float sun_emission_nits = 6.0e5f;
	};

	struct RendererSettings
	{
		int max_bounce_depth = 3;
		bool generate_bloom = false;
		bool use_karis_average = false;
		float bloom_blend = 0.3f;
		float bloom_internal_blend = 0.75f;
		bool enable_auto_exposure = false;
	};

	//Adheres to GLTF 2.0 specifications
	struct MaterialSceneEntity
	{
		MaterialSceneEntity() = default;

		MaterialSceneEntity(
			int albedo_texture_id,
			const glm::vec3& albedo,
			int orm_texture_id,
			float metallicity,
			float roughness,
			int transmission_texture_id,
			float transmission,
			float ior,
			int emission_texture_id,
			const glm::vec3& emission_factor,
			float emission_scale_nits,
			int normal_texture_id,
			float normal_scale)
			:
			albedo_texture_id(albedo_texture_id),
			albedo_factor(albedo),
			ORM_texture_id(orm_texture_id),
			metallic_factor(metallicity),
			roughness_factor(roughness),
			transmission_texture_id(transmission_texture_id),
			transmission_factor(transmission),
			ior(ior),
			emission_texture_id(emission_texture_id),
			emission_factor(emission_factor),
			emission_scale_nits(emission_scale_nits),
			normal_texture_id(normal_texture_id),
			normal_scale(normal_scale)
		{}
		//GLTF 2.0 spec: All empty texture reads must evaluate to 1.0f;

		int albedo_texture_id = -1;//MUST be srgb; decode before filtering
		glm::vec3 albedo_factor = { 1.0f,1.0f,1.0f };

		int ORM_texture_id = -1;//MUST be linear encoded
		float metallic_factor = 0.0f;
		float roughness_factor = 0.5f;

		int transmission_texture_id = -1;
		float transmission_factor = 0.0f;

		float ior = 1.45f;

		int emission_texture_id = -1;//MUST be srgb encoded; decode before use
		glm::vec3 emission_factor = { 1.0f,1.0f,1.0f };
		float emission_scale_nits = 0.0f;

		int normal_texture_id = -1;//MUST be linear; blue must be (0.5...1.0]=>(0.0f...1.0f); Tangent space
		float normal_scale = 1.0f;//scales X & Y

		bool isEmissive() const
		{
			glm::vec3 emit = emission_factor * emission_scale_nits;
			return (emit.r > 0.0f || emit.g > 0.0f || emit.b > 0.0f);
		};
	};

	struct TriangleSceneEntity
	{
		TriangleSceneEntity(
			glm::vec3 p0, glm::vec3 p1, glm::vec3 p2,
			glm::vec3 n0, glm::vec3 n1, glm::vec3 n2,
			glm::vec2 t0, glm::vec2 t1, glm::vec2 t2,
			int material_id) :
			p0(p0), p1(p1), p2(p2),
			n0(n0), n1(n1), n2(n2),
			t0(t0), t1(t1), t2(t2),
			material_id(material_id)
		{}

		glm::vec3 p0 = glm::vec3(0), p1 = glm::vec3(0), p2 = glm::vec3(0);
		glm::vec3 n0 = glm::vec3(0), n1 = glm::vec3(0), n2 = glm::vec3(0);
		glm::vec2 t0 = glm::vec2(0), t1 = glm::vec2(0), t2 = glm::vec2(0);
		int material_id = -1;

		float getArea() const
		{
			return 0.5f * length(cross(p1 - p0, p2 - p0));
		}
	};

	struct MeshSceneEntity
	{
		MeshSceneEntity(std::string_view name, const glm::mat4& model_transform) :
			name(name), model_matrix(model_transform) {}
		void addShape(const TriangleSceneEntity& shape)
		{
			shape_entities.push_back(shape);
		};

		glm::mat4 model_matrix;
		std::string name;
		std::vector<TriangleSceneEntity> shape_entities;
	};

	struct TextureSceneEntity
	{
		TextureSceneEntity(unsigned char* data, int width, int height, int chl_count)
			: pixels_data(data, data + width * height * chl_count),
			width(width), height(height), channels_count(chl_count)
		{}
		std::vector<unsigned char>pixels_data;
		int channels_count = 0;
		int width = 0, height = 0;
	};

	struct CameraSceneEntity
	{
		CameraSceneEntity(std::string_view name, glm::mat4 view_transform, float yfov_rads) :
			name(name), view_matrix(view_transform), y_fov_radians(yfov_rads) {}
		float y_fov_radians = 0.0f;
		glm::mat4 view_matrix;
		std::string name;
	};

	/// <summary>
	/// End product of a parsing/loading routine. Stores scene specification;
	/// Consumed by Renderer for instantiating internal scene;
	/// May implement "createXX()" methods to encapsulate entity to data conversion routines.
	/// </summary>
	struct BasicScene
	{
		void addTexture(const TextureSceneEntity& texture)
		{
			texture_entities.push_back(texture);
		}

		void addMaterial(const MaterialSceneEntity& material)
		{
			material_entities.push_back(material);
		};

		void addMesh(const MeshSceneEntity& msh)
		{
			mesh_entities.push_back(msh);
		};

		void addCamera(const CameraSceneEntity& cam)
		{
			camera_entities.push_back(cam);
		};

		//------------------
		std::vector<MaterialSceneEntity> material_entities;
		std::vector<MeshSceneEntity> mesh_entities;
		std::vector<CameraSceneEntity> camera_entities;
		std::vector<TextureSceneEntity> texture_entities;
	};
}/*KittlesPT*/