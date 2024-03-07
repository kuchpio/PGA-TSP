#include "SwapMutation.h"
#include <cuda_runtime.h>
#include <cstdlib>
#include <curand_kernel.h>

__device__ __host__ void SwapMutation::mutate(int* chromosome, int size, curandState* state) const
{
	int index1 = curand(state) % size;
	int index2 = curand(state) % size;
	std::swap(chromosome[index1], chromosome[index2]);
}