#include "material.cuh"
#include "containers.cuh"
#include "bsdf.cuh"
#include "sphere.cuh"

namespace KittlesPT
{
	KittlesPT::MaterialEvalContext::MaterialEvalContext(const SurfaceInteraction& surf)
	{
		wgnorm = surf.world_geometric_normal;
	}

	__device__ BSDF Material::getBSDF(const GlobalShaderData& shader_data, MaterialEvalContext ctx)
	{
	}
}