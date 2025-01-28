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
	float3 glm3_2f3(glm::vec3 v) {
		return make_float3(v.x, v.y, v.z);
	}

	float2 glm2_2f2(glm::vec2 v) {
		return make_float2(v.x, v.y);
	}
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

	//TODO: add API for direct content management
	struct RendererResource
	{
		thrust::universal_vector<TriangleMesh> scene_meshes;
		thrust::universal_vector<Triangle> scene_triangles;
		thrust::universal_vector<Light> scene_lights;
		thrust::universal_vector<Material> scene_materials;
		thrust::universal_vector<Texture> scene_textures;
		thrust::device_vector<float> histogram_buffer;

		thrust::device_vector<unsigned char> pixel_buffer;
		GlobalShaderData shader_global_data;
		std::unordered_map< std::string, TextureBuffer>m_frame_textures;
		MipChain bloom_mipchain;

		void destroy()
		{
			cudaFree(shader_global_data.scene_average_luminance);
			bloom_mipchain.destroy();

			for (std::pair<const std::string, TextureBuffer>& tex : m_frame_textures)
			{
				tex.second.destroy();
			}

			scene_triangles.clear();
			scene_lights.clear();
			scene_materials.clear();
			scene_textures.clear();
			pixel_buffer.clear();
			m_frame_textures.clear();
			histogram_buffer.clear();
		}

		~RendererResource()
		{
			destroy();
		}
	};

	void Renderer::init()
	{
		int cuda_driver_version, cuda_runtime_version;
		cudaDriverGetVersion(&cuda_driver_version); cudaRuntimeGetVersion(&cuda_runtime_version);
		std::printf("[RENDERER] CUDA driver version: %d.%d\n[RENDERER] CUDA toolkit runtime version: %d.%d\n",
			cuda_driver_version / 1000, cuda_driver_version % 100, cuda_runtime_version / 1000, cuda_runtime_version % 100);
		m_renderer_rsrc = new RendererResource();
		m_renderer_rsrc->m_frame_textures["main_texture"] = TextureBuffer();
		m_renderer_rsrc->m_frame_textures["gbuffer_texture"] = TextureBuffer();
		m_renderer_rsrc->m_frame_textures["accumulation_texture"] = TextureBuffer();
		m_renderer_rsrc->bloom_mipchain.init();
		m_renderer_rsrc->histogram_buffer = thrust::device_vector<float>((size_t)Constants::HISTOGRAM_SIZE, 0.0f);
		submitScene();

		//-------------------------

		cudaMallocManaged(&m_renderer_rsrc->shader_global_data.scene_average_luminance, sizeof(float));
		m_renderer_rsrc->shader_global_data.histogram_buffer = Buffer<float>(
			thrust::raw_pointer_cast(m_renderer_rsrc->histogram_buffer.data()), Constants::HISTOGRAM_SIZE);
		m_renderer_rsrc->shader_global_data.scene_camera = Camera(make_float3(0));
	}
	void Renderer::shutdown()
	{
		m_renderer_rsrc->destroy();
		delete m_renderer_rsrc;
		m_renderer_rsrc = nullptr;
	}

	void Renderer::resizeFrame(int width, int height)
	{
		if (m_width == width && m_height == height)
		{
			return;
		}
		m_width = width; m_height = height;
		m_renderer_rsrc->shader_global_data.frame_resolution = make_int2(m_width, m_height);
		glm::mat4 view = glm::mat4
		(1, 0, 0, 0,
			0, 1, 0, 0,
			0, 0, -1, 0,
			0, 0, 0, 1);

		//TODO: fix view
		setView(glm::perspectiveFovLH(glm::radians(90.0f),
			float(m_width), float(m_height), 1.f, 100.f),
			glm::inverse(view));

		for (std::pair<const std::string, TextureBuffer>& tex : m_renderer_rsrc->m_frame_textures)
		{
			if (tex.second.isInitialised())
			{
				tex.second.resize(m_width, m_height);
				continue;
			}
			std::printf("[RENDERER] Initializing renderer texture:%s\n", tex.first.c_str());
			tex.second.init(m_width, m_height);
		}

		m_renderer_rsrc->bloom_mipchain.resize(m_width, m_height);
	}

	float computeEV100(float average_luminance)
	{
		// K is a light meter calibration constant
		constexpr float K = 12.5f;
		return log2((average_luminance * 100.0f) / K);
	}

	// Notes:
	// EV below refers to EV at ISO 100

	// Given an aperture, shutter speed, and exposure value compute the required ISO value
	float ComputeISO(float aperture, float shutterSpeed, float ev)
	{
		return (Sqr(aperture) * 100.0f) / (shutterSpeed * pow(2.0f, ev));
	}

	// Given the camera settings compute the current exposure value
	float ComputeEV(float aperture, float shutterSpeed, float iso)
	{
		return log2((Sqr(aperture) * 100.0f) / (shutterSpeed * iso));
	}

	static float g_ev_comp = 0.0f;
	static Renderer::ExposureValues g_exposure_values;
	static float g_white_point = 0.0f;
	static float g_black_point = 0.0f;

	void applyAperturePriority(float focalLength,
		float targetEV,
		float& aperture,
		float& shutterSpeed,
		float& iso)
	{
		// Start with the assumption that we want a shutter speed of 1/f
		shutterSpeed = 1.0f / (focalLength * 1000.0f);

		// Compute the resulting ISO if we left the shutter speed here
		iso = clamp(ComputeISO(aperture, shutterSpeed, targetEV),
			static_cast<float>(g_exposure_values.MIN_ISO), static_cast<float>(g_exposure_values.MAX_ISO));
		// Figure out how far we were from the target exposure value
		float evDiff = targetEV - ComputeEV(aperture, shutterSpeed, iso);

		// Compute the final shutter speed
		shutterSpeed = clamp(shutterSpeed * pow(2.0f, -evDiff), g_exposure_values.MIN_SHUTTER_SECS,
			g_exposure_values.MAX_SHUTTER_SECS);
	}

	void Renderer::executeRendering(float delta_time_ms)
	{
		m_renderer_rsrc->shader_global_data.frame_delta = delta_time_ms;
		m_renderer_rsrc->shader_global_data.main_texture = m_renderer_rsrc->m_frame_textures["main_texture"].enableCudaAccess();
		m_renderer_rsrc->shader_global_data.accumulation_texture = m_renderer_rsrc->m_frame_textures["accumulation_texture"].enableCudaAccess();
		m_renderer_rsrc->shader_global_data.gbuffer_texture = m_renderer_rsrc->m_frame_textures["gbuffer_texture"].enableCudaAccess();

		launchPathTraceComputeMegaKernel(m_renderer_rsrc->shader_global_data);

		//generate bloom buffer
		if (m_renderer_rsrc->shader_global_data.renderer_settings.generate_bloom)
		{
			executeBloomGeneration();
			m_renderer_rsrc->shader_global_data.bloom_texture = m_renderer_rsrc->bloom_mipchain.mip_textures[0].enableCudaAccess();
		}

		//auto exposure pipeline
		if (m_renderer_rsrc->shader_global_data.renderer_settings.enable_auto_exposure)
		{
			launchHistogramComputeKernel(m_renderer_rsrc->shader_global_data);
			launchHistogramAverageComputeKernel(m_renderer_rsrc->shader_global_data);

			float avg_lum = *m_renderer_rsrc->shader_global_data.scene_average_luminance;
			float ev_target = computeEV100(avg_lum) - g_ev_comp;
			ev_target += 6.0f;//TODO:FIXME:Nasty fix for auto exposure undererestimation
			applyAperturePriority(0.1f, ev_target,
				g_exposure_values.aperture_f_num, g_exposure_values.shutter_speed_secs, g_exposure_values.ISO);
			setExposure(g_exposure_values, g_ev_comp, g_white_point, g_black_point);
			//printf("[delta %.3fms]: avg scene lm: %.3f\n", delta_time_ms, avg_lum);
		}

		launchPostProcessComputeKernel(m_renderer_rsrc->shader_global_data);

		if (m_renderer_rsrc->shader_global_data.renderer_settings.generate_bloom) {
			m_renderer_rsrc->bloom_mipchain.mip_textures[0].disableCudaAccess(m_renderer_rsrc->shader_global_data.bloom_texture);
		}

		m_renderer_rsrc->m_frame_textures["main_texture"].disableCudaAccess(m_renderer_rsrc->shader_global_data.main_texture);
		m_renderer_rsrc->m_frame_textures["accumulation_texture"].disableCudaAccess(m_renderer_rsrc->shader_global_data.accumulation_texture);
		m_renderer_rsrc->m_frame_textures["gbuffer_texture"].disableCudaAccess(m_renderer_rsrc->shader_global_data.gbuffer_texture);

		m_renderer_rsrc->shader_global_data.frame_index++;//TODO:expose to host as readonly?
	}

	void Renderer::getRenderTargetTexture(GLuint r_texture)
	{
		m_renderer_rsrc->m_frame_textures["main_texture"].copyTo(r_texture);
	}

	void Renderer::getDebugRenderTargetTexture(GLuint r_texture)
	{
		m_renderer_rsrc->bloom_mipchain.mip_textures[0].copyTo(r_texture);
	}

	bool Renderer::setMaterial(int idx, MaterialSceneEntity material)
	{
		if (idx >= m_renderer_rsrc->scene_materials.size()) {
			return false;
		}

		Material old_material = m_renderer_rsrc->scene_materials[idx];
		Material new_material(
			material.albedo_texture_id,
			glm3_2f3(material.albedo_factor),
			material.ORM_texture_id,
			material.metallic_factor,
			material.roughness_factor,
			material.transmission_texture_id,
			material.transmission_factor,
			material.ior,
			material.emission_texture_id,
			glm3_2f3(material.emission_factor),
			material.emission_scale_nits,
			material.normal_texture_id,
			material.normal_scale
		);
		m_renderer_rsrc->scene_materials[idx] = new_material;

		resetAccumulation();

		return true;
	}

	MaterialSceneEntity Renderer::getMaterial(int idx)
	{
		if (idx >= m_renderer_rsrc->scene_materials.size())
		{
			assert("OUT OF BOUNDS MATERIAL ACCESS");
		}

		Material mat = m_renderer_rsrc->scene_materials[idx];

		MaterialSceneEntity ret_mat(
			mat.albedo_texture_id, glm::vec3(mat.albedo.x, mat.albedo.y, mat.albedo.z),
			mat.ORM_texture_id, mat.metallic_factor, mat.roughness_factor,
			mat.transmission_texture_id, mat.transmission_factor,
			mat.ior,
			mat.emission_texture_id, glm::vec3(mat.emissive_factor.x, mat.emissive_factor.y, mat.emissive_factor.z), mat.emission_scale_nits,
			mat.normal_texture_id, mat.normal_scale
		);
		return ret_mat;
	}

	size_t Renderer::getMaterialsCount()
	{
		return m_renderer_rsrc->scene_materials.size();
	}

	void Renderer::setProceduralEnvironmentData(ProceduralEnvironmentData data)
	{
		m_renderer_rsrc->shader_global_data.procedural_environment_data = data;
		resetAccumulation();
	}

	ProceduralEnvironmentData Renderer::getProceduralEnvironmentData()
	{
		return m_renderer_rsrc->shader_global_data.procedural_environment_data;
	}

	void Renderer::setRendererSettings(const RendererSettings& cfg)
	{
		m_renderer_rsrc->shader_global_data.renderer_settings = cfg;
		resetAccumulation();
	}

	RendererSettings Renderer::getRendererSettings()
	{
		return m_renderer_rsrc->shader_global_data.renderer_settings;
	}

	float getSaturationBasedExposure(float aperture, float shutter_time, float iso)
	{
		//measuring for iso = S max
		constexpr float q = 0.65f;
		float l_max = (78.0f / q) * (Sqr(aperture) / (iso * shutter_time));
		return 1.0f / l_max;//why reciprocal?
	}

	float getStandardOutputBasedExposure(float aperture,
		float shutterSpeed,
		float iso,
		float middleGrey = 0.18f)
	{
		//for 18% gray derived from 118/255 after gamma correction
		constexpr float q = 0.65f;
		float l_avg = (10.0f / q) * (Sqr(aperture) / (iso * shutterSpeed));
		return middleGrey / l_avg;
	}

	void Renderer::setExposure(ExposureValues camera_values, float ev_comp, float white_point_ev, float black_point_ev)
	{
		/*
		* lens properties:
		* lens transmission(T)=0.9
		* vignettefactor(v(theta))=0.98(constant)
		* theta=10deg(angle from lens axis)
		* q = 0.65
		*/

		g_exposure_values = camera_values;
		g_ev_comp = ev_comp; g_white_point = white_point_ev; g_black_point = black_point_ev;

		float luminance_exposure_scalar = getStandardOutputBasedExposure(camera_values.aperture_f_num,
			camera_values.shutter_speed_secs, camera_values.ISO);
		m_renderer_rsrc->shader_global_data.scene_camera.setExposure(luminance_exposure_scalar,
			white_point_ev, black_point_ev);
	}

	Renderer::ExposureValues Renderer::getExposure()
	{
		return g_exposure_values;
	}

	void Renderer::resetAccumulation()
	{
		glClearTexImage(m_renderer_rsrc->m_frame_textures["accumulation_texture"].m_GL_texture, 0, GL_RGBA, GL_FLOAT, NULL);
		m_renderer_rsrc->shader_global_data.frame_index = 0;
	}

	void Renderer::setView(glm::mat4 projection_mat, glm::mat4 view_mat)
	{
		Mat4 proj = Mat4(projection_mat);
		Mat4 view = Mat4(view_mat);
		m_renderer_rsrc->shader_global_data.scene_camera.setView(proj.inverse(), view.inverse());

		resetAccumulation();
	}

	void Renderer::loadScene(const BasicScene& parsed_scene)
	{
		printf("starting textures\n");

		int bit_depth = 8;
		for (const TextureSceneEntity& tex : parsed_scene.texture_entities)
		{
			printf("tex: %d x %d | ch:%d\n", tex.width, tex.height, tex.channels_count);
			m_renderer_rsrc->scene_textures.push_back(
				Texture(tex.width, tex.height, tex.channels_count, bit_depth,
					(int)m_renderer_rsrc->pixel_buffer.size()));
			m_renderer_rsrc->pixel_buffer.insert(m_renderer_rsrc->pixel_buffer.end(),
				tex.pixels_data.begin(), tex.pixels_data.end());
		}

		printf("loaded %zu textures\nstarting materials\n", m_renderer_rsrc->scene_textures.size());

		for (const MaterialSceneEntity& mat : parsed_scene.material_entities)
		{
			m_renderer_rsrc->scene_materials.push_back(Material(
				mat.albedo_texture_id,
				glm3_2f3(mat.albedo_factor),
				mat.ORM_texture_id,
				mat.metallic_factor,
				mat.roughness_factor,
				mat.transmission_texture_id,
				mat.transmission_factor,
				mat.ior,
				mat.emission_texture_id,
				glm3_2f3(mat.emission_factor),
				mat.emission_scale_nits,
				mat.normal_texture_id,
				mat.normal_scale
			));
		}

		printf("loaded %zu materials\nstarting geometry\n", m_renderer_rsrc->scene_materials.size());

		for (const MeshSceneEntity& mesh : parsed_scene.mesh_entities)
		{
			size_t mesh_prim_start_id = m_renderer_rsrc->scene_triangles.size() - 1;
			for (const TriangleSceneEntity& tri : mesh.shape_entities)
			{
				const MaterialSceneEntity& mat = parsed_scene.material_entities[tri.material_id];
				int light_id = -1;

				if (mat.isEmissive())
				{
					int prim_id = (int)(m_renderer_rsrc->scene_triangles.size());
					m_renderer_rsrc->scene_lights.push_back(Light(tri.getArea(), prim_id, glm3_2f3(mat.emission_factor), mat.emission_scale_nits));
					light_id = (int)(m_renderer_rsrc->scene_lights.size() - 1);
				}

				m_renderer_rsrc->scene_triangles.push_back(Triangle(
					Vertex(glm3_2f3(tri.p0), glm3_2f3(tri.n0), glm2_2f2(tri.t0)),
					Vertex(glm3_2f3(tri.p1), glm3_2f3(tri.n1), glm2_2f2(tri.t1)),
					Vertex(glm3_2f3(tri.p2), glm3_2f3(tri.n2), glm2_2f2(tri.t2)),
					tri.material_id, light_id));
			}
			size_t mesh_prim_end_id = m_renderer_rsrc->scene_triangles.size() - 1;

			TriangleMesh tri_mesh(mesh_prim_start_id,
				mesh.shape_entities.size(), Mat4(glm::inverse(mesh.model_matrix)));
			m_renderer_rsrc->scene_meshes.push_back(tri_mesh);
		}
		std::printf("[RENDERER] loaded %zu shapes : %zu lights\n",
			m_renderer_rsrc->scene_triangles.size(), m_renderer_rsrc->scene_lights.size());

		submitScene();
	}

	void Renderer::submitScene()
	{
		m_renderer_rsrc->shader_global_data.meshes_buffer =
			Buffer<TriangleMesh>(
				thrust::raw_pointer_cast(m_renderer_rsrc->scene_meshes.data()),
				m_renderer_rsrc->scene_meshes.size());

		m_renderer_rsrc->shader_global_data.triangles_buffer =
			Buffer<Triangle>(
				thrust::raw_pointer_cast(m_renderer_rsrc->scene_triangles.data()),
				m_renderer_rsrc->scene_triangles.size());

		m_renderer_rsrc->shader_global_data.materials_buffer =
			Buffer<Material>(
				thrust::raw_pointer_cast(m_renderer_rsrc->scene_materials.data()),
				m_renderer_rsrc->scene_materials.size());

		m_renderer_rsrc->shader_global_data.lights_buffer =
			Buffer<Light>(
				thrust::raw_pointer_cast(m_renderer_rsrc->scene_lights.data()),
				m_renderer_rsrc->scene_lights.size());

		m_renderer_rsrc->shader_global_data.pixel_buffer =
			Buffer<unsigned char>(
				thrust::raw_pointer_cast(m_renderer_rsrc->pixel_buffer.data()),
				m_renderer_rsrc->pixel_buffer.size());

		m_renderer_rsrc->shader_global_data.texture_buffer =
			Buffer<Texture>(
				thrust::raw_pointer_cast(m_renderer_rsrc->scene_textures.data()),
				m_renderer_rsrc->scene_textures.size());
	}

	void Renderer::executeBloomGeneration()
	{
		//downscale
		for (int miplevel = 0; miplevel < m_renderer_rsrc->bloom_mipchain.max_mip_level; miplevel++)
		{
			TextureBuffer& src = m_renderer_rsrc->bloom_mipchain.mip_textures[miplevel];
			TextureBuffer& dst = m_renderer_rsrc->bloom_mipchain.mip_textures[miplevel + 1];

			if (miplevel == 0)
			{
				m_renderer_rsrc->m_frame_textures["main_texture"].disableCudaAccess(m_renderer_rsrc->shader_global_data.main_texture);
				m_renderer_rsrc->m_frame_textures["main_texture"].copyTo(src);
				m_renderer_rsrc->shader_global_data.main_texture = m_renderer_rsrc->m_frame_textures["main_texture"].enableCudaAccess();
			}
			DeviceTextureBuffer dsrc = src.enableCudaAccess();
			DeviceTextureBuffer ddst = dst.enableCudaAccess();

			launchBloomDownSampleComputeKernel(m_renderer_rsrc->shader_global_data, dsrc, ddst,
				(m_renderer_rsrc->shader_global_data.renderer_settings.use_karis_average && miplevel == 0));

			src.disableCudaAccess(dsrc);
			dst.disableCudaAccess(ddst);
		}
		//upscale
		for (int miplevel = m_renderer_rsrc->bloom_mipchain.max_mip_level; miplevel > 0; miplevel--)
		{
			TextureBuffer& src = m_renderer_rsrc->bloom_mipchain.mip_textures[miplevel];
			TextureBuffer& dst = m_renderer_rsrc->bloom_mipchain.mip_textures[miplevel - 1];

			DeviceTextureBuffer dsrc = src.enableCudaAccess();
			DeviceTextureBuffer ddst = dst.enableCudaAccess();

			launchBloomUpSampleComputeKernel(m_renderer_rsrc->shader_global_data, dsrc, ddst);

			src.disableCudaAccess(dsrc);
			dst.disableCudaAccess(ddst);
		}
	}
}