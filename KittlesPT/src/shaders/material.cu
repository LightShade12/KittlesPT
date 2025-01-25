#include "material.cuh"
#include "maths/vector_maths.cuh"
#include "containers.cuh"
#include "bsdf.cuh"
#include "interaction.cuh"

namespace KittlesPT
{
	__device__ MaterialEvalContext::MaterialEvalContext(const SurfaceInteraction& surf) :
		wgnorm(surf.world_geometric_normal), backface(surf.backface), uv(surf.uv), wpos(surf.world_position)
	{}

	//========================================================================================================

	__device__ BSDF Material::getBSDF(const GlobalShaderData& shader_data, MaterialEvalContext ctx) const
	{
		RGBSpectrum eval_albedo = RGBSpectrum(albedo);
		float eval_roughness = roughness_factor;
		float eval_metalness = metallic_factor;

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
		}
		if (normal_texture_id >= 0) {
			RGBSpectrum sampled = shader_data.texture_buffer.data[normal_texture_id].evaluate(shader_data, TextureEvalContext(ctx));
			float3 normal_encoded = powf(sampled, 2.2f).toFloat3();
			float3 mapped_normal = (normal_encoded * 2.0f) - 1.0f;
			mapped_normal.x *= normal_scale;
			mapped_normal.y *= normal_scale;
			mapped_normal.y = -mapped_normal.y;//DX12 => GL convention
			Mat3 frame = generateONBFrisvad(ctx.wgnorm);
			ctx.wgnorm = frame * normalize(mapped_normal);
		}

		//TODO: consider moving basis generation to BSDF constructor
		BSDF bsdf = BSDF(generateONBFrisvad(ctx.wgnorm),
			eval_albedo,
			eval_metalness,
			eval_roughness,
			transmission_factor,
			ior,
			ctx.backface);

		return bsdf;
	}
}/*KittlesPT*/