#ifndef __BASIC_MUTATION_H__
#define __BASIC_MUTATION_H__

#include <cuda_runtime.h>
#include <curand_kernel.h>

namespace tsp {

	template <typename gene>
	__device__ __forceinline__
	void mutate(gene* chromosome, unsigned int size, curandState* state)
	{
		unsigned int index1 = curand(state) % size;
		unsigned int index2 = curand(state) % size;
		gene tmp = chromosome[index1];
		chromosome[index1] = chromosome[index2];
		chromosome[index2] = tmp;
	}

}

#endif 
