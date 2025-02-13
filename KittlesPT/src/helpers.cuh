#pragma once
#include "glm/glm.hpp"
#include "maths/vector_types_extension.cuh"

namespace KittlesPT
{
	float3 glm3_2f3(glm::vec3 v) {
		return make_float3(v.x, v.y, v.z);
	}

	float2 glm2_2f2(glm::vec2 v) {
		return make_float2(v.x, v.y);
	}

	glm::vec3 f3_2glm3(float3 v) {
		return glm::vec3(v.x, v.y, v.z);
	}

	glm::vec2 f2_2glm2(float2 v) {
		return glm::vec2(v.x, v.y);
	}
}