#ifndef __INTERVAL_CROSSOVER_H__
#define __INTERVAL_CROSSOVER_H__

#include <cuda_runtime.h>
#include <curand_kernel.h>

namespace tsp {
	__device__
		bool containsVertex(int const* chromosome, int size, int vertex)
	{
		for (int i = 0; i < size; ++i)
		{
			if (chromosome[i] == vertex)
				return true;
		}
		return false;
	}

	__device__
		int* intervalCrossover(int* parent1, int* parent2, int size, curandState* state)
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
}

#endif
