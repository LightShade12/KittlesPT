#include "texture.cuh"
#include "color.cuh"
#include "interaction.cuh"
#include "material.cuh"

namespace KittlesPT
{
	//TODO: replace with CUDA texture memory

	__device__ TextureEvalContext::TextureEvalContext(const MaterialEvalContext& ctx) :
		uv(ctx.uv), wpos(ctx.wpos)
	{}

	//=================================================================================================================
}/*KittlesPT*/