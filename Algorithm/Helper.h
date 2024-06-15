#ifndef __HELPER_H__
#define __HELPER_H__

#define WARP_SIZE 32
#define FULL_MASK 0xffffffff

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>

namespace tsp {

	struct IslandGeneticAlgorithmOptions {
		unsigned int islandCount;
		unsigned int islandPopulationSize;
		unsigned int isolatedIterationCount;
		unsigned int migrationCount;
		float crossoverProbability;
		float mutationProbability;
		bool elitism;
		unsigned int stalledMigrationsLimit;
	};

	__global__
	void setupCurand(curandState* globalState, int seed) {
		int tid = blockIdx.x * blockDim.x + threadIdx.x;
		curand_init(seed, tid, 0, globalState + tid);
	}

	template <bool reduceMin = true, bool reduceMax = true>
	__device__ __forceinline__ void findMinMax(const unsigned int* array, unsigned int arraySize, unsigned int* reductionBuffer, 
		unsigned int minIndex, unsigned int maxIndex, unsigned int *minIndexOutput, unsigned int *maxIndexOutput) 
	{
		unsigned int wid = threadIdx.x / WARP_SIZE;			// Block warp id
		unsigned int lid = threadIdx.x & (WARP_SIZE - 1);	// Warp thread id

		unsigned int* reducedMin, * reducedMinIndex, * reducedMax, * reducedMaxIndex;
		unsigned int min, max;

		if (reduceMin) {
			reducedMin = reductionBuffer;
			reducedMinIndex = reducedMin + WARP_SIZE;
			min = 0xffffffff;
			minIndex = threadIdx.x;
		}
		if (reduceMax) {
			reducedMax = reductionBuffer + (reduceMin ? 2 * WARP_SIZE : 0);
			reducedMaxIndex = reducedMax + WARP_SIZE;
			max = 0;
			maxIndex = threadIdx.x;
		}

		// 1. Block stride loop thread reduction
		for (unsigned int chromosomeIndex = threadIdx.x; chromosomeIndex < arraySize; chromosomeIndex += blockDim.x) {
			unsigned int value = array[chromosomeIndex];
			if (reduceMin && min > value) {
				min = value;
				minIndex = chromosomeIndex;
			}
			if (reduceMax && max < value) {
				max = value;
				maxIndex = chromosomeIndex;
			}
		}

		// 2. Warp reduction in each warp
		unsigned int minShuf, minIndexShuf, maxShuf, maxIndexShuf;
		for (int i = 1; i < WARP_SIZE; i *= 2) {

			if (reduceMin) {
				minShuf = __shfl_xor_sync(FULL_MASK, min, i);
				minIndexShuf = __shfl_xor_sync(FULL_MASK, minIndex, i);
				if (min > minShuf) {
					minIndex = minIndexShuf;
					min = minShuf;
				}
			}

			if (reduceMax) {
				maxShuf = __shfl_xor_sync(FULL_MASK, max, i);
				maxIndexShuf = __shfl_xor_sync(FULL_MASK, maxIndex, i);
				if (max < maxShuf) {
					maxIndex = maxIndexShuf;
					max = maxShuf;
				}
			}

		}

		if (lid == 0) {
			if (reduceMin) {
				reducedMin[wid] = min;
				reducedMinIndex[wid] = minIndex;
			}
			if (reduceMax) {
				reducedMax[wid] = max;
				reducedMaxIndex[wid] = maxIndex;
			}
		}

		__syncthreads();

		// 3. Warp reduction of reduced results
		if (wid == 0) {

			if (reduceMin) {
				min = reducedMin[lid];
				minIndex = reducedMinIndex[lid];
				if (lid >= blockDim.x / WARP_SIZE) min = 0xffffffff;
			}
			if (reduceMax) {
				max = reducedMax[lid];
				maxIndex = reducedMaxIndex[lid];
				if (lid >= blockDim.x / WARP_SIZE) max = 0;
			}

			for (int i = 1; i < WARP_SIZE; i *= 2) {

				if (reduceMin) {
					minShuf = __shfl_xor_sync(FULL_MASK, min, i);
					minIndexShuf = __shfl_xor_sync(FULL_MASK, minIndex, i);
					if (min > minShuf) {
						minIndex = minIndexShuf;
						min = minShuf;
					}
				}

				if (reduceMax) {
					maxShuf = __shfl_xor_sync(FULL_MASK, max, i);
					maxIndexShuf = __shfl_xor_sync(FULL_MASK, maxIndex, i);
					if (max < maxShuf) {
						maxIndex = maxIndexShuf;
						max = maxShuf;
					}
				}

			}

			if (lid == 0) {
				if (minIndex == maxIndex) {
					*minIndexOutput = 0;
					*maxIndexOutput = arraySize - 1;
				} else {
					*minIndexOutput = minIndex;
					*maxIndexOutput = maxIndex;
				}
			}
		}
	}

}

#endif
