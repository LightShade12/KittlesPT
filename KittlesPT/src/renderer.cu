#include "renderer.hpp"

#include "maths/vector_maths.cuh"
#include "containers.cuh"
#include "shaders/kernels.cuh"

#include "glm/glm.hpp"
#include "glm/gtc/matrix_transform.hpp"

#include <thrust/universal_vector.h>
#include <thrust/device_vector.h>

#include <unordered_map>
#include <vector>
#include <string>
#include <iostream>

namespace KittlesPT
{
	//TODO:Add proper logging

	class MipChain
	{
	public:

		void init()
		{
			for (int mip_level = 0; mip_level < max_mip_count; mip_level++)
			{
				mip_textures.push_back(TextureBuffer());
			}
		}

		void resize(int base_width, int base_height)
		{
			max_mip_level = getMaxValidMipLevels({ base_width, base_height });
			max_mip_level = std::min(max_mip_level, max_mip_count - 1);

			for (int miplevel = 0; miplevel <= max_mip_level; miplevel++)
			{
				int mip_width = base_width >> miplevel;
				int	mip_height = base_height >> miplevel;

				TextureBuffer& mip_texture = mip_textures[miplevel];

				if (mip_texture.isInitialised()) {
					mip_texture.resize(mip_width, mip_height);
				}
				else {
					mip_texture.init(mip_width, mip_height);
				}
			}
		}

		void destroy()
		{
			for (TextureBuffer& tex : mip_textures)
			{
				tex.destroy();
			}
			mip_textures.clear();
		}

		//excludes mip0
		static int getMaxValidMipLevels(int2 t_base_resolution)
		{
			int mipx = static_cast<int>(std::log2(t_base_resolution.x)), mipy = static_cast<int>(std::log2(t_base_resolution.y));
			return std::min(mipx, mipy);
		}

	public:
		const int max_mip_count = 7;
		int max_mip_level = 0;
		std::vector<TextureBuffer> mip_textures;
	};

	struct RendererData
	{
		thrust::universal_vector<Sphere> scene_spheres;
		thrust::universal_vector<Light> scene_lights;
		thrust::universal_vector<Material> scene_materials;
		thrust::universal_vector<Texture> scene_textures;
		thrust::device_vector<unsigned char> pixel_buffer;
		GlobalShaderData shader_global_data;
		std::unordered_map< std::string, TextureBuffer>m_frame_textures;
		MipChain bloom_mipchain;

		void destroy()
		{
			bloom_mipchain.destroy();

			for (std::pair<const std::string, TextureBuffer>& tex : m_frame_textures)
			{
				tex.second.destroy();
			}

			scene_spheres.clear();
			scene_lights.clear();
			scene_materials.clear();
			scene_textures.clear();
			pixel_buffer.clear();
			m_frame_textures.clear();
		}

		~RendererData()
		{
			destroy();
		}
	};

	void Renderer::init()
	{
		int cuda_driver_version, cuda_runtime_version;
		cudaDriverGetVersion(&cuda_driver_version); cudaRuntimeGetVersion(&cuda_runtime_version);
		printf("[RENDERER] CUDA driver version: %d.%d\n[RENDERER] CUDA toolkit runtime version: %d.%d\n",
			cuda_driver_version / 1000, cuda_driver_version % 100, cuda_runtime_version / 1000, cuda_runtime_version % 100);
		m_renderer_data = new RendererData();
		m_renderer_data->m_frame_textures["main_texture"] = TextureBuffer();
		m_renderer_data->m_frame_textures["gbuffer_texture"] = TextureBuffer();
		m_renderer_data->m_frame_textures["accumulation_texture"] = TextureBuffer();
		m_renderer_data->bloom_mipchain.init();
		submitScene();

		//-------------------------

		m_renderer_data->shader_global_data.scene_camera = Camera(make_float3(0));
	}
	void Renderer::shutdown()
	{
		m_renderer_data->destroy();
		delete m_renderer_data;
		m_renderer_data = nullptr;
	}

	void Renderer::resizeFrame(int width, int height)
	{
		if (m_width == width && m_height == height)
		{
			return;
		}
		m_width = width; m_height = height;
		m_renderer_data->shader_global_data.frame_resolution = make_int2(m_width, m_height);
		glm::mat4 view = glm::mat4
		(1, 0, 0, 0,
			0, 1, 0, 0,
			0, 0, -1, 0,
			0, 0, 0, 1);

		//TODO: fix view
		setView(glm::perspectiveFovLH(glm::radians(90.0f),
			float(m_width), float(m_height), 1.f, 100.f),
			glm::inverse(view));

		for (std::pair<const std::string, TextureBuffer>& tex : m_renderer_data->m_frame_textures)
		{
			if (tex.second.isInitialised())
			{
				tex.second.resize(m_width, m_height);
				continue;
			}
			printf("Initializing texture:%s\n", tex.first.c_str());
			tex.second.init(m_width, m_height);
		}

		m_renderer_data->bloom_mipchain.resize(m_width, m_height);
	}
	void Renderer::executeRendering()
	{
		m_renderer_data->shader_global_data.main_texture = m_renderer_data->m_frame_textures["main_texture"].enableCudaAccess();
		m_renderer_data->shader_global_data.accumulation_texture = m_renderer_data->m_frame_textures["accumulation_texture"].enableCudaAccess();
		m_renderer_data->shader_global_data.gbuffer_texture = m_renderer_data->m_frame_textures["gbuffer_texture"].enableCudaAccess();

		launchPathTraceComputeKernel(m_renderer_data->shader_global_data);

		//generate bloom buffer
		if (m_renderer_data->shader_global_data.pathtracer_settings.generate_bloom)
		{
			executeBloomGeneration();
			m_renderer_data->shader_global_data.bloom_texture = m_renderer_data->bloom_mipchain.mip_textures[0].enableCudaAccess();
		}

		launchPostProcessComputeKernel(m_renderer_data->shader_global_data);

		if (m_renderer_data->shader_global_data.pathtracer_settings.generate_bloom) {
			m_renderer_data->bloom_mipchain.mip_textures[0].disableCudaAccess(m_renderer_data->shader_global_data.bloom_texture);
		}

		m_renderer_data->m_frame_textures["main_texture"].disableCudaAccess(m_renderer_data->shader_global_data.main_texture);
		m_renderer_data->m_frame_textures["accumulation_texture"].disableCudaAccess(m_renderer_data->shader_global_data.accumulation_texture);
		m_renderer_data->m_frame_textures["gbuffer_texture"].disableCudaAccess(m_renderer_data->shader_global_data.gbuffer_texture);

		m_renderer_data->shader_global_data.frame_index++;//TODO:expose to host as readonly?
	}

	void Renderer::getRenderTargetTexture(GLuint r_texture)
	{
		m_renderer_data->m_frame_textures["main_texture"].copyTo(r_texture);
	}

	void Renderer::getDebugRenderTargetTexture(GLuint r_texture)
	{
		m_renderer_data->bloom_mipchain.mip_textures[0].copyTo(r_texture);
	}

	bool Renderer::setMaterial(int idx, glm::vec3 albedo_factor, float metallicity, float roughness,
		float transmission, float ior)
	{
		if (idx >= m_renderer_data->scene_materials.size())
		{
			return false;
		}

		Material old_mat = m_renderer_data->scene_materials[idx];
		Material material(
			make_float3(albedo_factor.r, albedo_factor.g, albedo_factor.b),
			metallicity,
			roughness,
			transmission,
			ior,
			old_mat.emissive_factor,
			old_mat.emission_scale,
			old_mat.albedo_texture_id);
		m_renderer_data->scene_materials[idx] = material;

		resetAccumulation();

		return true;
	}

	bool Renderer::getMaterial(int idx, glm::vec3* albedo_factor, float* metallicity, float* roughness, float* transmission, float* ior)
	{
		if (idx >= m_renderer_data->scene_materials.size())
		{
			return false;
		}
		Material mat = m_renderer_data->scene_materials[idx];
		*albedo_factor = glm::vec3(mat.albedo.x, mat.albedo.y, mat.albedo.z);
		*metallicity = mat.metallicity;
		*roughness = mat.roughness;
		*transmission = mat.transmission;
		*ior = mat.ior;

		return true;
	}

	int Renderer::getMaterialsCount()
	{
		return (int)m_renderer_data->scene_materials.size();
	}

	void Renderer::setProceduralEnvironmentData(ProceduralEnvironmentData data)
	{
		m_renderer_data->shader_global_data.procedural_environment_data = data;
		resetAccumulation();
	}

	ProceduralEnvironmentData Renderer::getProceduralEnvironmentData()
	{
		return m_renderer_data->shader_global_data.procedural_environment_data;
	}

	void Renderer::setPathTracerSettings(PathtracerSettings cfg)
	{
		m_renderer_data->shader_global_data.pathtracer_settings = cfg;
		resetAccumulation();
	}

	PathtracerSettings Renderer::getPathTracerSettings()
	{
		return m_renderer_data->shader_global_data.pathtracer_settings;
	}

	float getSaturationBasedExposure(float aperture, float shutterSpeed, float iso)
	{
		float l_max = (7800.0f / 65.0f) * Sqr(aperture) / (iso * shutterSpeed);
		return 1.0f / l_max;
	}

	float getStandardOutputBasedExposure(float aperture,
		float shutterSpeed,
		float iso,
		float middleGrey = 0.18f)
	{
		float l_avg = (1000.0f / 65.0f) * Sqr(aperture) / (iso * shutterSpeed);
		return middleGrey / l_avg;
	}

	void Renderer::setExposure(float aperture_f_num, float shutter_speed_sec, float iso)
	{
		m_renderer_data->shader_global_data.scene_camera.film.exposure_EV = getSaturationBasedExposure(aperture_f_num, shutter_speed_sec, iso);
		resetAccumulation();
	}

	void Renderer::resetAccumulation()
	{
		glClearTexImage(m_renderer_data->m_frame_textures["accumulation_texture"].m_GL_texture, 0, GL_RGBA, GL_FLOAT, NULL);
		m_renderer_data->shader_global_data.frame_index = 0;
	}

	void Renderer::setView(glm::mat4 projection_mat, glm::mat4 view_mat)
	{
		//TODO: skip inversion
		Mat4 proj = Mat4(projection_mat);
		Mat4 view = Mat4(view_mat);
		m_renderer_data->shader_global_data.scene_camera.setView(proj.inverse(), view.inverse());

		resetAccumulation();
	}

	void Renderer::loadScene(const BasicScene& parsed_scene)
	{
		printf("starting textures\n");

		for (const TextureSceneEntity& tex : parsed_scene.texture_entities)
		{
			printf("tex: %d x %d | ch:%d\n", tex.width, tex.height, tex.channels_count);
			int bit_depth = 8;
			m_renderer_data->scene_textures.push_back(
				Texture(tex.width, tex.height, tex.channels_count, bit_depth,
					(int)m_renderer_data->pixel_buffer.size()));

			m_renderer_data->pixel_buffer.insert(m_renderer_data->pixel_buffer.end(),
				tex.pixels_data.begin(), tex.pixels_data.end());
		}

		printf("loaded %zu textures\nstarting materials\n", m_renderer_data->scene_textures.size());

		for (const MaterialSceneEntity& mat : parsed_scene.material_entities)
		{
			m_renderer_data->scene_materials.push_back(Material(
				make_float3(mat.albedo_factor.r, mat.albedo_factor.g, mat.albedo_factor.b),
				mat.metallicity,
				mat.roughness,
				mat.transmission,
				mat.ior,
				make_float3(mat.emission_factor.r, mat.emission_factor.g, mat.emission_factor.b),
				mat.emission_scale,
				mat.albedo_tex_id
			));
		}

		printf("loaded %zu materials\nstarting geometry\n", m_renderer_data->scene_materials.size());

		for (const SphereSceneEntity& sphere : parsed_scene.shape_entities)
		{
			const MaterialSceneEntity& sphere_mat = parsed_scene.material_entities[sphere.material_id];
			bool is_light = sphere_mat.isEmissive();
			int light_id = -1;

			if (is_light)
			{
				m_renderer_data->scene_lights.push_back(
					Light(sphere.getArea(),
						(int)(m_renderer_data->scene_spheres.size()),
						make_float3(sphere_mat.emission_factor.r, sphere_mat.emission_factor.g, sphere_mat.emission_factor.b),
						sphere_mat.emission_scale)
				);
				light_id = (int)(m_renderer_data->scene_lights.size() - 1);
			}

			m_renderer_data->scene_spheres.push_back(
				Sphere(sphere.radius,
					make_float3(sphere.position.x, sphere.position.y, sphere.position.z),
					sphere.material_id,
					light_id)
			);
		}
		printf("loaded %zu shapes : %zu lights\n",
			m_renderer_data->scene_spheres.size(),
			m_renderer_data->scene_lights.size());

		submitScene();
	}

	void Renderer::submitScene()
	{
		m_renderer_data->shader_global_data.geometry_buffer =
			Buffer<Sphere>(
				thrust::raw_pointer_cast(m_renderer_data->scene_spheres.data()),
				m_renderer_data->scene_spheres.size());

		m_renderer_data->shader_global_data.materials_buffer =
			Buffer<Material>(
				thrust::raw_pointer_cast(m_renderer_data->scene_materials.data()),
				m_renderer_data->scene_materials.size());

		m_renderer_data->shader_global_data.lights_buffer =
			Buffer<Light>(
				thrust::raw_pointer_cast(m_renderer_data->scene_lights.data()),
				m_renderer_data->scene_lights.size());

		m_renderer_data->shader_global_data.pixel_buffer =
			Buffer<unsigned char>(
				thrust::raw_pointer_cast(m_renderer_data->pixel_buffer.data()),
				m_renderer_data->pixel_buffer.size());

		m_renderer_data->shader_global_data.texture_buffer =
			Buffer<Texture>(
				thrust::raw_pointer_cast(m_renderer_data->scene_textures.data()),
				m_renderer_data->scene_textures.size());
	}

	void Renderer::executeBloomGeneration()
	{
		//downscale
		for (int miplevel = 0; miplevel < m_renderer_data->bloom_mipchain.max_mip_level; miplevel++)
		{
			TextureBuffer& src = m_renderer_data->bloom_mipchain.mip_textures[miplevel];
			TextureBuffer& dst = m_renderer_data->bloom_mipchain.mip_textures[miplevel + 1];

			if (miplevel == 0)
			{
				m_renderer_data->m_frame_textures["main_texture"].disableCudaAccess(m_renderer_data->shader_global_data.main_texture);
				m_renderer_data->m_frame_textures["main_texture"].copyTo(src);
				m_renderer_data->shader_global_data.main_texture = m_renderer_data->m_frame_textures["main_texture"].enableCudaAccess();
			}
			DeviceTextureBuffer dsrc = src.enableCudaAccess();
			DeviceTextureBuffer ddst = dst.enableCudaAccess();

			launchBloomDownSampleComputeKernel(m_renderer_data->shader_global_data, dsrc, ddst,
				(m_renderer_data->shader_global_data.pathtracer_settings.use_karis_average) ? (miplevel == 0) : false);

			src.disableCudaAccess(dsrc);
			dst.disableCudaAccess(ddst);
		}
		//upscale
		for (int miplevel = m_renderer_data->bloom_mipchain.max_mip_level; miplevel > 0; miplevel--)
		{
			TextureBuffer& src = m_renderer_data->bloom_mipchain.mip_textures[miplevel];
			TextureBuffer& dst = m_renderer_data->bloom_mipchain.mip_textures[miplevel - 1];

			DeviceTextureBuffer dsrc = src.enableCudaAccess();
			DeviceTextureBuffer ddst = dst.enableCudaAccess();

			launchBloomUpSampleComputeKernel(m_renderer_data->shader_global_data, dsrc, ddst);

			src.disableCudaAccess(dsrc);
			dst.disableCudaAccess(ddst);
		}
	}
}