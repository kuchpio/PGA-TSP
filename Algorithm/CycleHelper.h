#ifndef __CYCLE_HELPER_H__
#define __CYCLE_HELPER_H__

#define MAX_DISTANCE_CAN 1000000

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>

namespace tsp {

	template <typename vertex>
	__device__ __forceinline__ void initializeCycle(vertex* cycle, unsigned int size, curandState *state) 
	{
		// Initialize chromosome with a sequence from 0 to size - 1
		for (int i = 0; i < size; ++i) {
			cycle[i] = i;
		}
		
		// Fisher-Yates shuffle algorithm
		for (int i = size - 1; i > 0; i--) {
			int j = curand(state) % (i + 1);

			// Swap chromosome[i] with chromosome[j]
			int temp = cycle[i];
			cycle[i] = cycle[j];
			cycle[j] = temp;
		}
	}

	template <typename Instance, typename vertex>
	__device__ __forceinline__ unsigned int calculateCycleWeight(const vertex* cycle, const Instance instance)
	{
		int sum = edgeWeight(instance, cycle[size(instance) - 1], cycle[0]);

		for (int i = 0; i < size(instance) - 1; i++) {
			sum += edgeWeight(instance, cycle[i], cycle[i + 1]);
		}

		return MAX_DISTANCE_CAN - sum;
	}

}

#endif
