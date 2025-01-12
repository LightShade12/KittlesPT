#pragma once
#include "glm/glm.hpp"
#include "glm/gtc/constants.hpp"
#include <vector>

namespace KittlesPT
{
	struct ProceduralEnvironmentData
	{
		float sun_angular_diameter_rad = 0.0087f;
		float sun_phi_rad = 4.18879f;
		//float sun_theta_rad = 0.785f;
		float sun_theta_rad = 0.0872665f;
		float sun_radiance_intensity = 50.0f;//def: 50.0f
	};

	struct PathtracerSettings
	{
		int max_bounce_depth = 3;
		bool generate_bloom = false;
		bool use_karis_average = false;
	};

	struct MaterialSceneEntity
	{
		MaterialSceneEntity(const glm::vec3& albedo,
			float metallicity,
			float roughness,
			float transmission,
			float ior,
			const glm::vec3& emission,
			float emission_scale,
			int albedo_tex_id)
			:
			albedo_factor(albedo),
			metallicity(metallicity),
			roughness(roughness),
			transmission(transmission),
			ior(ior),
			emission_factor(emission),
			emission_scale(emission_scale),
			albedo_tex_id(albedo_tex_id)
		{}

		glm::vec3 albedo_factor = glm::vec3(1);
		float metallicity = 0.0f;
		float roughness = 0.5f;
		float transmission = 0.0f;
		float ior = 1.45f;
		glm::vec3 emission_factor = glm::vec3(1);
		float emission_scale = 0.0f;
		int albedo_tex_id = -1;

		bool isEmissive() const
		{
			glm::vec3 emit = emission_factor * emission_scale;
			return (emit.r > 0.0f || emit.g > 0.0f || emit.b > 0.0f);
		};
	};

	struct SphereSceneEntity
	{
		SphereSceneEntity(float radius,
			const glm::vec3& position,
			int material_id) :
			radius(radius),
			position(position),
			material_id(material_id)
		{}

		float radius;
		glm::vec3 position = glm::vec3(0);
		int material_id = 0;

		float getArea() const
		{
			return 4.0f * glm::pi<float>() * (radius * radius);
		}
	};

	class TextureSceneEntity
	{
	public:

		TextureSceneEntity(unsigned char* data, int width, int height, int chl_count)
			: pixels_data(data, data + width * height * chl_count),
			width(width), height(height), channels_count(chl_count)
		{}
		std::vector<unsigned char>pixels_data;
		int channels_count = 0;
		int width = 0, height = 0;
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

		void addMaterial(MaterialSceneEntity material)
		{
			material_entities.push_back(material);
		};
		void addShape(SphereSceneEntity shape)
		{
			shape_entities.push_back(shape);
		};

		//------------------
		std::vector<MaterialSceneEntity> material_entities;
		std::vector<SphereSceneEntity> shape_entities;
		std::vector<TextureSceneEntity> texture_entities;
	};
}