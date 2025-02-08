#include "material.cuh"
#include "maths/vector_maths.cuh"
#include "triangle.cuh"
#include "containers.cuh"
#include "bsdf.cuh"
#include "interaction.cuh"

namespace KittlesPT
{
	__device__ MaterialEvalContext::MaterialEvalContext(const SurfaceInteraction& surface) :
		wgnorm(surface.world_geometric_normal),
		backface(surface.backface),
		uv(surface.uv),
		wpos(surface.world_position),
		wo(surface.wo)
	{}
	__device__ BSDF Material::getBSDF(const ShaderData& shader_data, MaterialEvalContext ctx) const
	{
		RGBSpectrum eval_albedo = RGBSpectrum(albedo);
		float eval_roughness = roughness_factor;
		float eval_metalness = metallic_factor;
		float eval_transmission = transmission_factor;

		if (albedo_texture_id >= 0)
		{
			RGBSpectrum sampled = shader_data.texture_buffer.data[albedo_texture_id].evaluate(shader_data, TextureEvalContext(ctx));
			sampled = powf(sampled, 2.2f);//sRGB to linear approx
			eval_albedo *= sampled;
		}
		if (ORM_texture_id >= 0) {
			RGBSpectrum sampled = shader_data.texture_buffer.data[ORM_texture_id].evaluate(shader_data, TextureEvalContext(ctx));
			sampled = powf(sampled, 2.2f);//sRGB to linear approx
			//ORM: r=Occlusion, g=Roughness, b=Metalness
			eval_roughness *= sampled.g;
			eval_metalness *= sampled.b;
		}
		if (transmission_texture_id >= 0)
		{
			RGBSpectrum sampled = shader_data.texture_buffer.data[transmission_texture_id].evaluate(shader_data, TextureEvalContext(ctx));
			sampled = powf(sampled, 2.2f);//sRGB to linear approx
			eval_transmission *= sampled.r;
		}
		//normal map application
		if (normal_texture_id >= 0) {
			RGBSpectrum sampled = shader_data.texture_buffer.data[normal_texture_id].evaluate(shader_data, TextureEvalContext(ctx));
			float3 normal_encoded = powf(sampled, 2.2f).toFloat3();
			float3 mapped_normal = (normal_encoded * 2.0f) - 1.0f;
			mapped_normal.x *= normal_scale;
			mapped_normal.y *= normal_scale;
			//mapped_normal.y = -mapped_normal.y;//DX12 => GL convention
			Mat3 frame = generateONBFrisvad(ctx.wgnorm);
			float3 shading_wn = frame * normalize(mapped_normal);

			ctx.wgnorm = (dot(ctx.wo, shading_wn) > 0.0f) ? shading_wn : ctx.wgnorm;
		}

		//TODO: consider moving basis generation to BSDF constructor
		BSDF bsdf = BSDF(generateONBFrisvad(ctx.wgnorm),
			eval_albedo,
			eval_metalness,
			eval_roughness,
			eval_transmission,
			ior,
			ctx.backface);

		return bsdf;
	}
}/*KittlesPT*/