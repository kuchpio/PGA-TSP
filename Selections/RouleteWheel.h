#ifndef __ROULETTE_WHEEL_SELECTION_H__
#define __ROULETTE_WHEEL_SELECTION_H__

#include <cuda_runtime.h>
#include <curand_kernel.h>

namespace tsp {
	__device__
		int rouletteWheelSelection(int* fitness, int populationSize, curandState* state, float totalFitness) {
		int slice = (int)(curand_uniform(state) * totalFitness);
		int total = 0;
		for (int i = blockDim.x * blockIdx.x; i < blockDim.x * (blockIdx.x + 1); ++i) {
			total += fitness[i];
			if (total > slice) {
				return i;
			}
		}
		return populationSize - 1;
	}
}

#endif
