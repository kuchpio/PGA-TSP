#ifndef __BASIC_CROSSOVER_H__
#define __BASIC_CROSSOVER_H__

#include <cuda_runtime.h>
#include <curand_kernel.h>

namespace tsp {
	// Partially Mapped Crossover (PMX)
	__device__
		void PMX(const int* parent1, const int* parent2, int* offspring1, int* offspring2, const int size, curandState* state) {
		int cutPoint1 = curand(state) % size;
		int cutPoint2 = curand(state) % size;

		if (cutPoint1 > cutPoint2) {
			int temp = cutPoint1;
			cutPoint1 = cutPoint2;
			cutPoint2 = temp;
		}

		// Copy the segments between cut points
		for (int i = cutPoint1; i <= cutPoint2; i++) {
			offspring1[i] = parent2[i];
			offspring2[i] = parent1[i];
		}

		// Mapping remainder elements outside the cut points
		for (int i = 0; i < size; i++) {
			if (i < cutPoint1 || i > cutPoint2) {
				offspring1[i] = parent1[i];
				offspring2[i] = parent2[i];
			}
			// Resolve conflicts
			for (int j = cutPoint1; j <= cutPoint2; j++) {
				if (offspring1[i] == offspring1[j] && i != j) {
					offspring1[i] = parent2[i];
				}
				if (offspring2[i] == offspring2[j] && i != j) {
					offspring2[i] = parent1[i];
				}
			}
		}
	}
}

#endif
