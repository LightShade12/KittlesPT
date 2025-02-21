#include "renderer.hpp"

#include "maths/vector_maths.cuh"
#include "containers.cuh"
#include "as_builder.cuh"
#include "shaders/kernels.cuh"
#include "helpers.cuh"
#include "renderer_resource.cuh"

#include "glm/glm.hpp"
#include "glm/gtc/matrix_transform.hpp"

#include <vector>
#include <string>
#include <iostream>

namespace KittlesPT
{
	//TODO:Add proper logging
	//TODO: convert rsrc to rsrc manager

	void Renderer::initialize()
	{
		int cuda_driver_version, cuda_runtime_version;
		cudaDriverGetVersion(&cuda_driver_version); cudaRuntimeGetVersion(&cuda_runtime_version);
		std::printf("[RENDERER] CUDA driver version: %d.%d\n[RENDERER] CUDA toolkit runtime version: %d.%d\n",
			cuda_driver_version / 1000, cuda_driver_version % 100, cuda_runtime_version / 1000, cuda_runtime_version % 100);

		m_renderer_rsrc = new RendererResource();
		m_renderer_rsrc->bloom_mipchain.init();

		m_renderer_rsrc->m_frame_textures["output_texture"] = TextureBuffer();
		m_renderer_rsrc->m_frame_textures["render_texture"] = TextureBuffer();
		m_renderer_rsrc->m_frame_textures["gbuffer_texture"] = TextureBuffer();
		m_renderer_rsrc->m_frame_textures["prev_gbuffer_texture"] = TextureBuffer();
		m_renderer_rsrc->m_frame_textures["vbuffer_texture"] = TextureBuffer();
		m_renderer_rsrc->m_frame_textures["accumulation_texture"] = TextureBuffer();
		m_renderer_rsrc->m_frame_textures["debug_texture"] = TextureBuffer();
		m_renderer_rsrc->m_frame_textures["backbuffer_texture"] = TextureBuffer();

		m_renderer_rsrc->histogram_buffer = thrust::device_vector<float>(Constants::HISTOGRAM_SIZE, 0.0f);
		m_renderer_rsrc->shader_data.histogram_buffer = Buffer<float>(thrust::raw_pointer_cast(m_renderer_rsrc->histogram_buffer.data()), Constants::HISTOGRAM_SIZE);
		cudaMallocManaged(&m_renderer_rsrc->shader_data.scene_average_luminance, sizeof(float));

		m_renderer_rsrc->updateResource();
	}

	void Renderer::shutdown()
	{
		m_renderer_rsrc->destroy();
		delete m_renderer_rsrc;
		m_renderer_rsrc = nullptr;
	}

	void Renderer::resizeResolution(uint32_t width, uint32_t height)
	{
		bool upscale = m_renderer_rsrc->shader_data.renderer_settings.upscale_enable;
		float scale_factor = 2;

		if (m_output_width == width && m_output_height == height
			&& m_render_width == ((upscale) ? int(m_output_width / scale_factor) : m_output_width)
			&& m_render_height == ((upscale) ? int(m_output_height / scale_factor) : m_output_height))
		{
			return;
		}
		//printf("resizing; upscale %d \n", upscale);

		m_output_width = width, m_output_height = height;
		m_render_width = m_output_width, m_render_height = m_output_height;
		if (upscale) {
			m_render_width = int(m_render_width / scale_factor),
				m_render_height = int(m_render_height / scale_factor);
		}

		m_renderer_rsrc->shader_data.output_resolution = make_int2(m_output_width, m_output_height);
		m_renderer_rsrc->shader_data.render_resolution = make_int2(m_render_width, m_render_height);

		//recompute projection for new screen size
		glm::mat4 old_proj = m_renderer_rsrc->shader_data.scene_camera.curr_inv_projection_matrix.inverse().toGLM();
		float fov_rad = 2.0f * atan(1.0f / old_proj[1][1]);
		glm::mat4 projection = glm::perspectiveFovLH(fov_rad, (float)m_render_width, (float)m_render_height,
			0.1f, 100.0f);
		m_renderer_rsrc->shader_data.scene_camera.curr_inv_projection_matrix = Mat4(projection).inverse();
		resetAccumulation();

		for (std::pair<const std::string, TextureBuffer>& tex : m_renderer_rsrc->m_frame_textures)
		{
			bool is_output_res = (tex.first == "output_texture") || (tex.first == "backbuffer_texture");

			if (tex.second.isInitialised()) {
				tex.second.resize((is_output_res) ? m_output_width : m_render_width,
					(is_output_res) ? m_output_height : m_render_height);
				continue;
			}
			std::printf("[RENDERER] Initializing renderer texture:%s\n", tex.first.c_str());
			tex.second.initialize((is_output_res) ? m_output_width : m_render_width,
				(is_output_res) ? m_output_height : m_render_height);
		}

		m_renderer_rsrc->bloom_mipchain.resize(m_output_width, m_output_height);
	}

	void Renderer::executeRendering(float delta_time_ms)
	{
		m_renderer_rsrc->shader_data.frame_delta_ms = delta_time_ms;
		m_renderer_rsrc->shader_data.output_texture = m_renderer_rsrc->m_frame_textures["output_texture"].enableCudaAccess();
		m_renderer_rsrc->shader_data.render_texture = m_renderer_rsrc->m_frame_textures["render_texture"].enableCudaAccess();
		m_renderer_rsrc->shader_data.backbuffer_texture = m_renderer_rsrc->m_frame_textures["backbuffer_texture"].enableCudaAccess();
		m_renderer_rsrc->shader_data.accumulation_texture = m_renderer_rsrc->m_frame_textures["accumulation_texture"].enableCudaAccess();
		m_renderer_rsrc->shader_data.gbuffer_texture = m_renderer_rsrc->m_frame_textures["gbuffer_texture"].enableCudaAccess();
		m_renderer_rsrc->shader_data.prev_gbuffer_texture = m_renderer_rsrc->m_frame_textures["prev_gbuffer_texture"].enableCudaAccess();
		m_renderer_rsrc->shader_data.vbuffer_texture = m_renderer_rsrc->m_frame_textures["vbuffer_texture"].enableCudaAccess();
		m_renderer_rsrc->shader_data.debug_texture = m_renderer_rsrc->m_frame_textures["debug_texture"].enableCudaAccess();

		m_renderer_rsrc->updateTLAS();

		launchPathTraceComputeMegaKernel(m_renderer_rsrc->shader_data);
		if (m_renderer_rsrc->shader_data.renderer_settings.integrator_use_temporal_accumulation)
		{//update accumulation texture for next frame
			m_renderer_rsrc->m_frame_textures["render_texture"].disableCudaAccess(m_renderer_rsrc->shader_data.render_texture);
			m_renderer_rsrc->m_frame_textures["accumulation_texture"].disableCudaAccess(m_renderer_rsrc->shader_data.accumulation_texture);
			m_renderer_rsrc->m_frame_textures["render_texture"].copyTo(m_renderer_rsrc->m_frame_textures["accumulation_texture"]);
			m_renderer_rsrc->shader_data.accumulation_texture = m_renderer_rsrc->m_frame_textures["accumulation_texture"].enableCudaAccess();
			m_renderer_rsrc->shader_data.render_texture = m_renderer_rsrc->m_frame_textures["render_texture"].enableCudaAccess();
		}
		launchModulateComputeKernel(m_renderer_rsrc->shader_data);

		//auto exposure pipeline
		if (m_renderer_rsrc->shader_data.renderer_settings.tonemapper_enable_auto_exposure) {
			launchHistogramComputeKernel(m_renderer_rsrc->shader_data);
			launchHistogramAverageComputeKernel(m_renderer_rsrc->shader_data);
			AutoExposureProgram& ae = m_renderer_rsrc->auto_exposure_program; float avg_lum = *m_renderer_rsrc->shader_data.scene_average_luminance;
			ae.computeExposure(avg_lum);
			setExposure(ae.getExposureValues(), ae.getEVComp(), ae.getWhitePoint(), ae.getBlackPoint());
		}

		//UPSCALE HERE------------------
		if (m_renderer_rsrc->shader_data.renderer_settings.upscale_enable) {
			launchUpscaleComputeKernel(m_renderer_rsrc->shader_data.render_texture, m_renderer_rsrc->shader_data.output_texture,
				m_renderer_rsrc->shader_data.backbuffer_texture);

#ifdef USE_FSR
			m_renderer_rsrc->m_frame_textures["backbuffer_texture"].disableCudaAccess(m_renderer_rsrc->shader_data.backbuffer_texture);
			m_renderer_rsrc->m_frame_textures["output_texture"].disableCudaAccess(m_renderer_rsrc->shader_data.output_texture);
			m_renderer_rsrc->m_frame_textures["backbuffer_texture"].copyTo(m_renderer_rsrc->m_frame_textures["output_texture"]);
			m_renderer_rsrc->shader_data.output_texture = m_renderer_rsrc->m_frame_textures["output_texture"].enableCudaAccess();
			m_renderer_rsrc->shader_data.backbuffer_texture = m_renderer_rsrc->m_frame_textures["backbuffer_texture"].enableCudaAccess();
#endif
		}
		else {
			m_renderer_rsrc->m_frame_textures["render_texture"].disableCudaAccess(m_renderer_rsrc->shader_data.render_texture);
			m_renderer_rsrc->m_frame_textures["output_texture"].disableCudaAccess(m_renderer_rsrc->shader_data.output_texture);
			m_renderer_rsrc->m_frame_textures["render_texture"].copyTo(m_renderer_rsrc->m_frame_textures["output_texture"]);
			m_renderer_rsrc->shader_data.output_texture = m_renderer_rsrc->m_frame_textures["output_texture"].enableCudaAccess();
			m_renderer_rsrc->shader_data.render_texture = m_renderer_rsrc->m_frame_textures["render_texture"].enableCudaAccess();
		}

		//generate bloom buffer
		if (m_renderer_rsrc->shader_data.renderer_settings.bloom_generate_bloom) {
			executeBloomGeneration();
			m_renderer_rsrc->shader_data.bloom_texture = m_renderer_rsrc->bloom_mipchain.mip_textures[0].enableCudaAccess();
		}

		launchPostProcessComputeKernel(m_renderer_rsrc->shader_data);

		if (m_renderer_rsrc->shader_data.renderer_settings.postprocess_enable_effects) {
			launchFxComputeKernel(m_renderer_rsrc->shader_data);
		}
		else {
			m_renderer_rsrc->m_frame_textures["backbuffer_texture"].copyTo(m_renderer_rsrc->m_frame_textures["output_texture"]);
		}

		if (m_renderer_rsrc->shader_data.renderer_settings.bloom_generate_bloom) {
			m_renderer_rsrc->bloom_mipchain.mip_textures[0].disableCudaAccess(m_renderer_rsrc->shader_data.bloom_texture);
		}

		m_renderer_rsrc->m_frame_textures["output_texture"].disableCudaAccess(m_renderer_rsrc->shader_data.output_texture);
		m_renderer_rsrc->m_frame_textures["render_texture"].disableCudaAccess(m_renderer_rsrc->shader_data.render_texture);
		m_renderer_rsrc->m_frame_textures["backbuffer_texture"].disableCudaAccess(m_renderer_rsrc->shader_data.backbuffer_texture);
		m_renderer_rsrc->m_frame_textures["accumulation_texture"].disableCudaAccess(m_renderer_rsrc->shader_data.accumulation_texture);
		m_renderer_rsrc->m_frame_textures["gbuffer_texture"].disableCudaAccess(m_renderer_rsrc->shader_data.gbuffer_texture);
		m_renderer_rsrc->m_frame_textures["prev_gbuffer_texture"].disableCudaAccess(m_renderer_rsrc->shader_data.prev_gbuffer_texture);
		m_renderer_rsrc->m_frame_textures["vbuffer_texture"].disableCudaAccess(m_renderer_rsrc->shader_data.vbuffer_texture);
		m_renderer_rsrc->m_frame_textures["debug_texture"].disableCudaAccess(m_renderer_rsrc->shader_data.debug_texture);

		m_renderer_rsrc->m_frame_textures["gbuffer_texture"].copyTo(m_renderer_rsrc->m_frame_textures["prev_gbuffer_texture"]);

		//updating camera matices per frame
		{
			const Camera& cam = m_renderer_rsrc->shader_data.scene_camera;
			m_renderer_rsrc->shader_data.scene_camera.setView(cam.curr_inv_projection_matrix, cam.curr_inv_view_matrix);
		}

		m_renderer_rsrc->shader_data.frame_index++;//TODO:expose to host as readonly?
	}

	void Renderer::getRenderTargetTexture(GLuint r_texture) const
	{
		m_renderer_rsrc->m_frame_textures["output_texture"].copyTo(r_texture);
	}

	void Renderer::getDebugRenderTargetTexture(GLuint r_texture) const
	{
		m_renderer_rsrc->m_frame_textures["debug_texture"].copyTo(r_texture);
	}

	bool Renderer::setMaterial(uint32_t idx, const MaterialSceneEntity& material)
	{
		if (idx >= m_renderer_rsrc->scene_materials.size()) {
			return false;
		}

		Material new_material(
			material.albedo_texture_id, glm3_2f3(material.albedo_factor),
			material.ORM_texture_id, material.metallic_factor, material.roughness_factor,
			material.transmission_texture_id, material.transmission_factor,
			material.ior,
			material.emission_texture_id, glm3_2f3(material.emission_factor), material.emission_scale_nits,
			material.normal_texture_id, material.normal_scale
		);
		m_renderer_rsrc->scene_materials[idx] = new_material;

		resetAccumulation();

		return true;
	}

	MaterialSceneEntity Renderer::getMaterial(uint32_t idx) const
	{
		if (idx >= m_renderer_rsrc->scene_materials.size())
		{
			assert("OUT OF BOUNDS MATERIAL ACCESS");
		}

		Material mat = m_renderer_rsrc->scene_materials[idx];

		MaterialSceneEntity ret_mat(
			"unpreserved_name",
			mat.albedo_texture_id, f3_2glm3(mat.albedo),
			mat.ORM_texture_id, mat.metallic_factor, mat.roughness_factor,
			mat.transmission_texture_id, mat.transmission_factor,
			mat.ior,
			mat.emission_texture_id, f3_2glm3(mat.emissive_factor), mat.emission_scale_nits,
			mat.normal_texture_id, mat.normal_scale
		);

		return ret_mat;
	}

	bool Renderer::setMeshTransform(uint32_t idx, const glm::mat4& model)
	{
		if (idx >= m_renderer_rsrc->scene_meshes.size()) {
			return false;
		}

		Mat4 model_mat(model);
		Mat4 inv_model_mat(glm::inverse(model));

		TriangleMesh mesh = m_renderer_rsrc->scene_meshes[idx];
		mesh.setTransform(inv_model_mat);
		m_renderer_rsrc->scene_meshes[idx] = mesh;
		m_renderer_rsrc->blas_buffer[mesh.blas_index].setTransform(model_mat);

		if (!m_renderer_rsrc->shader_data.renderer_settings.integrator_use_temporal_accumulation) {
			resetAccumulation();
		}

		return true;
	}

	glm::mat4 Renderer::getMeshTransform(uint32_t idx) const
	{
		if (idx >= m_renderer_rsrc->scene_meshes.size()) {
			assert("OUT OF BOUNDES ACCES[MESHES]");
		}
		glm::mat4 inv_model = m_renderer_rsrc->scene_meshes[idx].curr_inv_model_matrix.toGLM();
		return glm::inverse(inv_model);
	}

	size_t Renderer::getMaterialsCount() const
	{
		return m_renderer_rsrc->scene_materials.size();
	}

	size_t Renderer::getMeshCount() const
	{
		return m_renderer_rsrc->scene_meshes.size();
	}

	void Renderer::setProceduralEnvironmentData(const ProceduralEnvironmentSettings& data)
	{
		m_renderer_rsrc->shader_data.procedural_environment_data = data;
		resetAccumulation();
	}

	ProceduralEnvironmentSettings Renderer::getProceduralEnvironmentData() const
	{
		return m_renderer_rsrc->shader_data.procedural_environment_data;
	}

	void Renderer::setRendererSettings(const RendererSettings& settings)
	{
		m_renderer_rsrc->shader_data.renderer_settings = settings;
		resetAccumulation();
	}

	RendererSettings Renderer::getRendererSettings() const
	{
		return m_renderer_rsrc->shader_data.renderer_settings;
	}

	void Renderer::setExposure(const ExposureValues& camera_values, float ev_comp, float white_point_ev, float black_point_ev)
	{
		/*
		* lens properties:
		* lens transmission(T)=0.9
		* vignettefactor(v(theta))=0.98(constant)
		* theta=10deg(angle from lens axis)
		* q = 0.65
		*/

		m_renderer_rsrc->auto_exposure_program.recordValues(camera_values, ev_comp,
			white_point_ev, black_point_ev);

		float luminance_exposure_scalar = AutoExposureProgram::getStandardOutputBasedExposure(camera_values.aperture_f_num,
			camera_values.shutter_speed_secs, camera_values.ISO);
		m_renderer_rsrc->shader_data.scene_camera.setExposure(luminance_exposure_scalar,
			white_point_ev, black_point_ev);
	}

	ExposureValues Renderer::getExposure() const
	{
		return m_renderer_rsrc->auto_exposure_program.getExposureValues();
	}

	void Renderer::resetAccumulation()
	{
		glClearTexImage(m_renderer_rsrc->m_frame_textures["accumulation_texture"].getGLTexture(), 0, GL_RGBA, GL_FLOAT, NULL);
		m_renderer_rsrc->shader_data.frame_index = 0;
	}

	void Renderer::setView(const glm::mat4& projection_matrix, const glm::mat4& view_matrix)
	{
		Mat4 projection(projection_matrix);
		Mat4 view(view_matrix);
		Mat4 inv_projection(glm::inverse(projection_matrix));
		Mat4 inv_view(glm::inverse(view_matrix));

		m_renderer_rsrc->shader_data.scene_camera.setView(inv_projection, inv_view);

		if (!m_renderer_rsrc->shader_data.renderer_settings.integrator_use_temporal_accumulation) {
			resetAccumulation();
		}
	}

	void Renderer::loadScene(const BasicScene& parsed_scene)
	{
		printf("starting textures\n");

		int32_t bit_depth = 8;
		for (const TextureSceneEntity& tex : parsed_scene.texture_entities)
		{
			printf("tex: %d x %d | ch:%d\n", tex.width, tex.height, tex.channels_count);
			m_renderer_rsrc->scene_textures.push_back(Texture(tex.width, tex.height, tex.channels_count, bit_depth,
				static_cast<int32_t>(m_renderer_rsrc->pixel_buffer.size())));
			m_renderer_rsrc->pixel_buffer.insert(m_renderer_rsrc->pixel_buffer.end(),
				tex.pixels_data.begin(), tex.pixels_data.end());
		}

		printf("loaded %zu textures\nstarting materials\n", m_renderer_rsrc->scene_textures.size());

		for (const MaterialSceneEntity& mat : parsed_scene.material_entities)
		{
			m_renderer_rsrc->scene_materials.push_back(Material(
				mat.albedo_texture_id, glm3_2f3(mat.albedo_factor),
				mat.ORM_texture_id, mat.metallic_factor, mat.roughness_factor,
				mat.transmission_texture_id, mat.transmission_factor,
				mat.ior,
				mat.emission_texture_id, glm3_2f3(mat.emission_factor), mat.emission_scale_nits,
				mat.normal_texture_id, mat.normal_scale
			));
		}

		printf("loaded %zu materials\nstarting geometry\n", m_renderer_rsrc->scene_materials.size());

		for (const MeshSceneEntity& mesh : parsed_scene.mesh_entities)
		{
			size_t mesh_prim_start_id = m_renderer_rsrc->scene_triangles.size();
			int32_t mesh_id = m_renderer_rsrc->scene_meshes.size();

			for (const TriangleSceneEntity& tri : mesh.shape_entities)
			{
				const MaterialSceneEntity& mat = parsed_scene.material_entities[tri.material_id];
				int32_t light_id = -1;

				if (mat.isEmissive())
				{
					int32_t prim_id = static_cast<int32_t>(m_renderer_rsrc->scene_triangles.size());
					m_renderer_rsrc->scene_lights.push_back(Light(tri.getArea(), prim_id, glm3_2f3(mat.emission_factor), mat.emission_scale_nits));
					light_id = static_cast<int32_t>(m_renderer_rsrc->scene_lights.size() - 1);
				}

				m_renderer_rsrc->scene_triangles.push_back(Triangle(
					Vertex(glm3_2f3(tri.p0), glm3_2f3(tri.n0), glm2_2f2(tri.t0)),
					Vertex(glm3_2f3(tri.p1), glm3_2f3(tri.n1), glm2_2f2(tri.t1)),
					Vertex(glm3_2f3(tri.p2), glm3_2f3(tri.n2), glm2_2f2(tri.t2)),
					tri.material_id, light_id, mesh_id));
			}

			TriangleMesh tri_mesh(static_cast<int32_t>(mesh_prim_start_id),
				static_cast<int32_t>(mesh.shape_entities.size()),
				Mat4(glm::inverse(mesh.model_matrix)));
			tri_mesh.blas_index = static_cast<int32_t>(m_renderer_rsrc->blas_buffer.size());//TODO:move to constructor

			m_renderer_rsrc->blas_buffer.push_back(m_renderer_rsrc->blas_builder.build(tri_mesh, mesh_id));
			m_renderer_rsrc->scene_meshes.push_back(tri_mesh);
		}

		m_renderer_rsrc->updateTLAS();

		std::printf("[RENDERER] loaded %zu shapes : %zu lights\n",
			m_renderer_rsrc->scene_triangles.size(), m_renderer_rsrc->scene_lights.size());

		if (!parsed_scene.camera_entities.empty()) {
			Camera new_camera;
			const CameraSceneEntity& parsed_camera = parsed_scene.camera_entities[0];
			new_camera.curr_inv_view_matrix = Mat4(glm::inverse(parsed_camera.view_matrix));
			new_camera.curr_inv_view_matrix[2] *= -1.0f;
			new_camera.curr_inv_projection_matrix = Mat4(glm::inverse(glm::perspectiveFovLH(parsed_camera.y_fov_radians,
				100.0f, 100.0f, 1.0f, 100.0f)));//Done to intialize reusable fovyrad in inv_proj_mat
			new_camera.curr_world_position = make_float3(new_camera.curr_inv_view_matrix[3]);
			m_renderer_rsrc->shader_data.scene_camera = new_camera;
		}

		m_renderer_rsrc->updateResource();
	}

	void Renderer::executeBloomGeneration()
	{
		//downscale
		for (uint32_t miplevel = 0; miplevel < m_renderer_rsrc->bloom_mipchain.max_mip_level; miplevel++)
		{
			TextureBuffer& src = m_renderer_rsrc->bloom_mipchain.mip_textures[miplevel];
			TextureBuffer& dst = m_renderer_rsrc->bloom_mipchain.mip_textures[miplevel + 1];

			if (miplevel == 0)
			{
				m_renderer_rsrc->m_frame_textures["output_texture"].disableCudaAccess(m_renderer_rsrc->shader_data.output_texture);
				m_renderer_rsrc->m_frame_textures["output_texture"].copyTo(src);
				m_renderer_rsrc->shader_data.output_texture = m_renderer_rsrc->m_frame_textures["output_texture"].enableCudaAccess();
			}
			DeviceTextureBuffer dsrc = src.enableCudaAccess();
			DeviceTextureBuffer ddst = dst.enableCudaAccess();

			launchBloomDownSampleComputeKernel(m_renderer_rsrc->shader_data, dsrc, ddst,
				(m_renderer_rsrc->shader_data.renderer_settings.bloom_use_karis_average && miplevel == 0));

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

			launchBloomUpSampleComputeKernel(m_renderer_rsrc->shader_data, dsrc, ddst);

			src.disableCudaAccess(dsrc);
			dst.disableCudaAccess(ddst);
		}
	}
}