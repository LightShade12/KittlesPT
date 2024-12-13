#pragma once
#include <glad/glad.h>
#include <cuda_runtime.h>

void check_cuda(cudaError_t result, char const* const func, const char* const file, int const line);

#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )
void check_cuda(cudaError_t result, char const* const func, const char* const file, int const line);

const char* glErrorString(GLenum err);