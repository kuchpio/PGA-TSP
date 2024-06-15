#ifndef __WARP_ROULETTE_WHEEL_SELECTION_H__
#define __WARP_ROULETTE_WHEEL_SELECTION_H__

#define WARP_SIZE 32
#define FULL_MASK 0xffffffff

#include <cuda_runtime.h>
#include <curand_kernel.h>

namespace tsp {

	__device__ __forceinline__ float createRoulletteWheel(unsigned int islandBestIndex, unsigned int islandWorstIndex, 
		unsigned int islandPopulationSize, unsigned int *cycleWeight, float *roulletteWheelThreshold) 
	{
		float *reducedRoulletteWheelThresholdSum = roulletteWheelThreshold + islandPopulationSize;

		unsigned int blockWid = threadIdx.x / WARP_SIZE;			// Block warp id
		unsigned int lid = threadIdx.x & (WARP_SIZE - 1);			// Warp thread id

		unsigned int min = cycleWeight[islandBestIndex];
		unsigned int max = cycleWeight[islandWorstIndex];

		float intervalWidthSum = 0.0f;
		for (unsigned int baseOffset = blockWid * WARP_SIZE; baseOffset < islandPopulationSize; baseOffset += blockDim.x) {
			float intervalWidth = 0.0f, intervalWidthShfl;
			if (baseOffset + lid < islandPopulationSize)
				intervalWidth = min == max ? 1.0f : ((float)(max - cycleWeight[baseOffset + lid])) / ((float)(max - min));

			// Warp scan to get partial prefix sums
			for (unsigned int i = 1; i < WARP_SIZE; i *= 2) {
				intervalWidthShfl = __shfl_up_sync(FULL_MASK, intervalWidth, i);
				if (lid >= i) intervalWidth += intervalWidthShfl;
			}

			if (baseOffset + lid < islandPopulationSize)
				roulletteWheelThreshold[baseOffset + lid] = intervalWidth;

			if (lid == WARP_SIZE - 1)
				intervalWidthSum += intervalWidth;
		}

		if (lid == WARP_SIZE - 1) {
			reducedRoulletteWheelThresholdSum[blockWid] = intervalWidthSum;
		}

		__syncthreads();

		if (blockWid == 0) {
			intervalWidthSum = reducedRoulletteWheelThresholdSum[lid];
			if (lid >= blockDim.x / WARP_SIZE) intervalWidthSum = 0.0f;
			for (unsigned int i = 1; i < WARP_SIZE; i *= 2) 
				intervalWidthSum += __shfl_xor_sync(FULL_MASK, intervalWidthSum, i);
			reducedRoulletteWheelThresholdSum[lid] = intervalWidthSum;
		}

		__syncthreads();

		return reducedRoulletteWheelThresholdSum[0];
	}
	
	__device__ __forceinline__ unsigned int selectIndex(curandState *localState, float *roulletteWheelThreshold, float maxThreshold, unsigned int islandPopulationSize) 
	{
		unsigned int lid = threadIdx.x & (WARP_SIZE - 1);			// Warp thread id

		float rnd;
		if (lid == 0) rnd = curand_uniform(localState) * maxThreshold;
		rnd = __shfl_sync(FULL_MASK, rnd, 0);

		for (unsigned int baseOffset = 0; baseOffset < islandPopulationSize; baseOffset += WARP_SIZE) {

			float threshold = baseOffset + lid < islandPopulationSize ? 
				roulletteWheelThreshold[baseOffset + lid] : roulletteWheelThreshold[islandPopulationSize - 1];

			unsigned int thresholdLid = __clz(~(__ballot_sync(FULL_MASK, rnd <= threshold)));
			if (thresholdLid) return baseOffset + WARP_SIZE - thresholdLid;

			rnd -= __shfl_sync(FULL_MASK, threshold, WARP_SIZE - 1);
		}
		
		return islandPopulationSize - 1; // Unreachable
	}

}

#endif
