#include "IntervalCrossover.h"
#include <cuda_runtime.h>
#include <cstdlib>
#include <curand_kernel.h>

__device__ __host__ int* IntervalCrossover::crossover(int* parent1, int* parent2, int size, curandState* state) const
{
	int left = curand(state) % (size - 1);
	int right = left + curand(state) % (size - left);
	int* child = new int[size] { -1 };
	for (int i = left; i <= right; ++i)
	{
		child[i] = parent1[i];
	}

	int emptyIndx = 0;
	for (int i = 0; i < size; ++i)
	{
		if (!containsVertex(child, size, i))
		{
			if (emptyIndx >= left && emptyIndx <= right)
			{
				emptyIndx = right + 1;
			}
			child[emptyIndx++] = i;
		}
	}

	return child;
}