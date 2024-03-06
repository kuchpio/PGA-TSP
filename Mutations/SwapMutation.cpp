#include "SwapMutation.h"
#include <cuda_runtime.h>
#include <cstdlib>

__device__ __host__ void SwapMutation::mutate(int* chromosome, int size) const
{
	int index1 = rand() % size;
	int index2 = rand() % size;
	std::swap(chromosome[index1], chromosome[index2]);
}
