#include <cuda_runtime.h>
#include <cstdlib>
#include <curand_kernel.h>
#include "device_launch_parameters.h"
#include "./RandomGenerator.cuh"

__global__ void setupCurand(curandState* state, unsigned long seed, int size)
{
	int id = threadIdx.x + blockDim.x * blockIdx.x;
	if (id < size)
	{
		curand_init(seed, id, 0, &state[id]);
	}
}