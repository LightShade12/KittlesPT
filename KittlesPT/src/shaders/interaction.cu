#include "interaction.cuh"
#include "containers.cuh"
#include "light.cuh"
#include "triangle.cuh"
#include "ray.cuh"
#include "bsdf.cuh"
#include "maths/constants.cuh"

namespace KittlesPT
{
	__device__ SurfaceInteraction Intersection::getSurfaceInteraction(const ShaderData& shader_data, const Ray& ray)
	{
		SurfaceInteraction surfintr;
		surfintr.primitive_id = primitive_id;
		surfintr.instance_id = instance_id;
		surfintr.distance = distance;
		surfintr.wo = -ray.getDirection();

		const Triangle& tri = shader_data.triangles_buffer.data[primitive_id];
		Mat4 model_mat = shader_data.meshes_buffer.data[instance_id].curr_inv_model_matrix.inverse();

		surfintr.material_id = tri.material_id;

		surfintr.world_position = ray.getPointAt(distance);
		surfintr.world_geometric_normal = normalize(make_float3(model_mat * make_float4(tri.local_geometric_normal, 0)));
		surfintr.uv = (bary_coords.x * tri.vertex0.tex_coords) + (bary_coords.y * tri.vertex1.tex_coords) + (bary_coords.z * tri.vertex2.tex_coords);

		if (!sameHemisphere(surfintr.world_geometric_normal, surfintr.wo)) {
			surfintr.world_geometric_normal *= -1.0f;
			surfintr.backface = true;
		}

		if (tri.light_id >= 0) {
			surfintr.arealight = &(shader_data.lights_buffer.data[tri.light_id]);
		}

		return surfintr;
	}

	//=========================================================================================

	__device__ RGBSpectrum SurfaceInteraction::Le(const ShaderData& shader_data, const Ray& ray) const
	{
		return(arealight) ? arealight->L(shader_data, uv) : RGBSpectrum(0.0f);
	}

	__device__ BSDF SurfaceInteraction::getBSDF(const ShaderData& shader_data) const
	{
		const Material& mat = shader_data.materials_buffer.data[material_id];

		BSDF bsdf = mat.getBSDF(shader_data, MaterialEvalContext(*this));

		return bsdf;
	}
}/*KittlesPT*/