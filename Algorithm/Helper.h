#ifndef __HELPER_H__
#define __HELPER_H__

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>

namespace tsp {

	__global__
	void setupCurand(curandState* globalState, int seed) {
		int tid = blockIdx.x * blockDim.x + threadIdx.x;
		curand_init(seed, tid, 0, globalState + tid);
	}

	__device__ 
	void initChromosome(int* chromosome, int size, curandState* state) {
		// Initialize chromosome with a sequence from 0 to size - 1
		for (int i = 0; i < size; ++i) {
			chromosome[i] = i;
		}
		
		// Fisher-Yates shuffle algorithm
		for (int i = size - 1; i > 0; i--) {
			int j = curand(state) % (i + 1);

			// Swap chromosome[i] with chromosome[j]
			int temp = chromosome[i];
			chromosome[i] = chromosome[j];
			chromosome[j] = temp;
		}
	}

}

#endif
