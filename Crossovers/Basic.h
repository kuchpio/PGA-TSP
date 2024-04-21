#ifndef __BASIC_CROSSOVER_H__
#define __BASIC_CROSSOVER_H__

#include <cuda_runtime.h>
#include <curand_kernel.h>

namespace tsp {
	__device__
		void crossover(int* parent1, int* parent2, int* child, int size, int fitness1, int fitness2) {
		if (fitness1 > fitness2)
		{
			for (int i = 0; i < size; i++) {
				child[i] = parent1[i];
			}
		}
		else
		{
			for (int i = 0; i < size; i++) {
				child[i] = parent2[i];
			}
		}
	}
}

#endif
