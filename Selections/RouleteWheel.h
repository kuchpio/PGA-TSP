#ifndef __ROULETTE_WHEEL_SELECTION_H__
#define __ROULETTE_WHEEL_SELECTION_H__

#include <cuda_runtime.h>
#include <curand_kernel.h>

namespace tsp {

	__device__
	int rouletteWheelSelection(float* fitness, int populationSize, curandState* state, float totalFitness) {
		float slice = curand_uniform(state) * totalFitness;
		float total = 0;
		for (int i = 0; i < populationSize; ++i) {
			total += fitness[i];
			if (total > slice) {
				return i;
			}
		}
		return populationSize - 1;
	}

}

#endif 
