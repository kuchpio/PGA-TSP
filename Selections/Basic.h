#ifndef __BASIC_SELECTION_H__
#define __BASIC_SELECTION_H__

#include <cuda_runtime.h>

namespace tsp {

	__device__
	void select(const int* chromosome1, const int* chromosome2, int* resultChromosome, int size, float fitness1, float fitness2) {
		if (fitness1 <= fitness2) {
			for (int i = 0; i < size; ++i) resultChromosome[i] = chromosome1[i];
		}
		else {
			for (int i = 0; i < size; ++i) resultChromosome[i] = chromosome2[i];
		}
	}

}

#endif 
