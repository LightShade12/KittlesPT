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
		if (albedo_texture_id >= 0)
		{
			/*if (albedo_texture_id == 2) {
				ctx.uv *= 1.5f;
			}
			else if (albedo_texture_id == 3)
			{
				ctx.uv *= 100.0f;
			}*/
			RGBSpectrum sampled = shader_data.texture_buffer.data[albedo_texture_id].evaluate(shader_data, TextureEvalContext(ctx));
			sampled = powf(sampled, 2.2f);//srgb to linear approx
			eval_albedo *= sampled;
		}

		//TODO: consider moving basis generation to BSDF constructor
		BSDF bsdf = BSDF(generateONBFrisvad(ctx.wgnorm),
			eval_albedo,
			metallic_factor,
			roughness_factor,
			transmission_factor,
			ior,
			ctx.backface);

		return bsdf;
	}
}/*KittlesPT*/