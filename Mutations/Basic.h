#ifndef __BASIC_MUTATION_H__
#define __BASIC_MUTATION_H__

#include <cuda_runtime.h>
#include <curand_kernel.h>

namespace tsp {

	__device__ 
	void mutate(int* chromosome, int size, curandState* state)
	{
		int index1 = curand(state) % size;
		int index2 = curand(state) % size;
		int tmp = chromosome[index1];
		chromosome[index1] = chromosome[index2];
		chromosome[index2] = tmp;
	}

}



#endif 
