#ifndef __BLOCK_SWAP_H__
#define __BLOCK_SWAP_H__

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/extrema.h>
#include <thrust/device_ptr.h>

namespace tsp {
	template <typename Instance>
	__global__ void ChangeBestChromosomes(const Instance instance, int* population, int* bestFitnessIndex, curandState* globalState) {
		int bid = blockIdx.x;
		int tid = threadIdx.x;
		__shared__ int indx;
		int instanceSize = size(instance);
		curandState localState = globalState[tid];

		if (tid == 0) {
			indx = bid * blockDim.x + curand(&localState) % blockDim.x;
		}
		__syncthreads();

		int blockStartIndex = indx * instanceSize;
		int bestChromosomeIndex = bestFitnessIndex[(bid + 1) % gridDim.x];
		int bestChromosomeStartIndex = bestChromosomeIndex * instanceSize;

		if (blockStartIndex != bestChromosomeStartIndex) {
			for (int i = tid; i < instanceSize; i += blockDim.x) {
				population[blockStartIndex + i] = population[bestChromosomeStartIndex + i];
			}
		}
	}
}

#endif