#ifndef __INTERVAL_CROSSOVER_H__
#define __INTERVAL_CROSSOVER_H__

#define WARP_SIZE 32

#include <cuda_runtime.h>
#include <curand_kernel.h>

namespace tsp {

	template<typename gene>
	__device__ __forceinline__
	void crossover(gene* chromosomeA, gene* chromosomeB, unsigned int n, curandState* state)
	{
		unsigned int lid = (blockDim.x * blockIdx.x + threadIdx.x) & (WARP_SIZE - 1);

		for (unsigned int i = lid; i < n; i += WARP_SIZE) {
			gene buffer = chromosomeA[i];
			chromosomeA[i] = chromosomeB[i];
			chromosomeB[i] = buffer;
		}

	}

}

#endif
