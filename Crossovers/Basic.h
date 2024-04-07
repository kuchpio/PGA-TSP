#ifndef __BASIC_CROSSOVER_H__
#define __BASIC_CROSSOVER_H__

#include <cuda_runtime.h>
#include <curand_kernel.h>

namespace tsp {

	__device__ 
	void crossover(int* parent1, int* parent2, int* child, int size, curandState* state) {
		for (int i = 0; i < size; i++) {
			child[i] = parent1[i];
		}
	}

}

#endif 
