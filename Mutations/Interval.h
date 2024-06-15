#ifndef __INTERVAL_MUTATION_H__
#define __INTERVAL_MUTATION_H__

#include <cuda_runtime.h>
#include <curand_kernel.h>

namespace tsp {
	__device__
		void intervalMutate(int* chromosome, int size, curandState* state)
	{
		int start = curand(state) % size;
		int end = curand(state) % size;
		if (start > end)
		{
			int temp = start;
			start = end;
			end = temp;
		}

		while (start < end)
		{
			int temp = chromosome[start];
			chromosome[start] = chromosome[end];
			chromosome[end] = temp;
			start++;
			end--;
		}
	}
}

#endif
