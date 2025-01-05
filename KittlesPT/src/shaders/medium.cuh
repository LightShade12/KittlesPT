#pragma once
#include "maths/linear_algebra.cuh"
#include "maths/constants.cuh"
#include "ray.cuh"

namespace KittlesPT
{
	struct PhaseFunctionSample
	{
		PhaseFunctionSample() = default;
		__device__ PhaseFunctionSample(float p, float3 wi, float pdf) :
			p(p), wi(wi), pdf(pdf) {};

		float p;//phase func eval
		float3 wi;
		float pdf;
	};

	class HGPhaseFunction 
	{
	public:
		HGPhaseFunction() = default;

	private:
		__device__ float henyeyGreenstein(float cosTheta, float g)
		{
			float denom = 1 + Sqr(g) + 2 * g * cosTheta;
			return (1.0f / (4.0f * Constants::PI)) * (1 - Sqr(g)) / (denom * sqrtf(denom));
		}

		__device__ float3 sampleHenyeyGreenstein(float3 wo, float g, float2 u, float* pdf)
		{
			float cosTheta;

			if (fabsf(g) < 1e-3f) {
				cosTheta = 1 - 2 * u.x;
			}
			else {
				cosTheta = -1 / (2 * g) *
					(1 + Sqr(g) - Sqr((1 - Sqr(g)) / (1 + g - 2 * g * u.y)));
			}

			float phi = 2 * Constants::PI * u.y;
			Mat3 frame = generateONBFrisvad(wo);
			float3 wi = frame * sphericalToCartesian(acosf(cosTheta), phi);

			if (pdf) {
				*pdf = henyeyGreenstein(cosTheta, g);
			}

			return wi;
		}

	public:

		__device__ float p(float3 wi, float3 wo)
		{
			return henyeyGreenstein(dot(wi, wo), g);
		}

		__device__ PhaseFunctionSample sample(float3 wo, float2 u2)
		{
			float pdf;
			float3 wi = sampleHenyeyGreenstein(wo, g, u2, &pdf);
			return PhaseFunctionSample(pdf, wi, pdf);
		}

		__device__ float pdf(float3 wo, float3 wi)
		{
			return p(wo, wi);
		}

	public:
		float g = 0.0f;
	};

	struct MediumProperties
	{
		float sigma_a, sigma_s;//TODO: maybe can be made RGBSpectrum for albedo
		HGPhaseFunction phase;
	};

	struct RayMajorantSegment
	{
		float tMin, tMax;
		float sigma_maj;
	};

	class HomogeneousMajorantIterator
	{
	public:
		RayMajorantSegment next()
		{
		};
	};

	class HomogeniousMedium
	{
	public:

		HomogeneousMajorantIterator sampleRay(const Ray& ray, float tmax)
		{
		}

		__device__ MediumProperties samplePoint(float3 p)
		{
		}

	public:
		float sigma_a_spec, sigma_s_spec;
		HGPhaseFunction phase;
	};

	struct MediumInterface
	{
	public:
		__device__ MediumInterface(HomogeniousMedium medium) :
			inside(medium), outside(medium) {};

		__device__ MediumInterface(HomogeniousMedium inside, HomogeniousMedium outside) :
			inside(inside), outside(outside) {};

		bool isMediumTransition();

	public:
		HomogeniousMedium inside, outside;
	};
}