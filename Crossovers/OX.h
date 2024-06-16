#ifndef __OX_CROSSOVER_H__
#define __OX_CROSSOVER_H__

#include <cuda_runtime.h>
#include <curand_kernel.h>

namespace tsp {
	__device__
		void OX(int* const parent1, int* const parent2, int* child, int size, curandState* state)
	{
		int left = curand(state) % (size - 1);
		int right = left + curand(state) % (size - left);
		for (int i = 0; i < size; ++i)
		{
			child[i] = i >= left && i <= right ? parent1[i] : -1;
		}

		int emptyIndx = 0;
		for (int i = 0; i < size; ++i)
		{
			bool alreadyInChild = false;
			for (int j = left; j <= right; ++j) {
				if (child[j] == parent2[i]) {
					alreadyInChild = true;
					break;
				}
			}

			if (!alreadyInChild) {
				if (emptyIndx >= left && emptyIndx <= right) {
					emptyIndx = right + 1;
				}
				if (emptyIndx < size) {
					child[emptyIndx++] = parent2[i];
				}
			}
		}
	}
}

#endif
