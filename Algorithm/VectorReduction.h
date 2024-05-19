#ifndef __VECTORREDUCTION_H__
#define __VECTORREDUCTION_H__

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>

namespace tsp
{
	__device__ void SumVectorForChromosomes(int* totalFitness, int* fitness, int* sharedFitness)
	{
		int tid = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
		int bid = threadIdx.x;
		totalFitness[bid] = fitness[tid] + fitness[tid + 1];
		sharedFitness[bid] = fitness[tid] > fitness[tid + 1] ? fitness[tid] : fitness[tid + 1];
		__syncthreads();
		for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
			if (bid < stride) {
				if (sharedFitness[bid] < sharedFitness[bid + stride]) {
					sharedFitness[bid] = sharedFitness[bid + stride];
				}
				totalFitness[bid] += totalFitness[bid + stride];
			}
			__syncthreads();
		}
		__syncthreads();
	}

	__device__ void SumVector(int* totalFitness, int* fitness)
	{
		int tid = blockIdx.x * blockDim.x + threadIdx.x;
		int bid = threadIdx.x;
		totalFitness[bid] = fitness[tid] + fitness[tid + 1];
		__syncthreads();
		for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
			if (bid < stride) {
				totalFitness[bid] += totalFitness[bid + stride];
			}
			__syncthreads();
		}
	}

	__device__ void SumAndGetMaxVector(int* totalFitness, int* fitness, int* sharedFitness, int* sharedIndexes)
	{
		int tid = blockIdx.x * blockDim.x + threadIdx.x;
		int bid = threadIdx.x;
		sharedFitness[bid] = fitness[tid];
		sharedIndexes[bid] = bid;
		totalFitness[bid] = fitness[tid];
		__syncthreads();
		for (int stride = blockDim.x >> 2; stride > 0; stride >>= 1) {
			if (bid < stride) {
				if (sharedFitness[bid] < sharedFitness[bid + stride]) {
					sharedFitness[bid] = sharedFitness[bid + stride];
					sharedIndexes[bid] = sharedIndexes[bid + stride];
				}
				totalFitness[bid] += totalFitness[bid + stride];
			}
			__syncthreads();
		}
	}
}

#endif