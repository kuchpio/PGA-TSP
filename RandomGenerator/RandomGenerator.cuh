#pragma once
#include <curand_kernel.h>

__global__ void setupCurand(curandState* state, unsigned long seed, int size);