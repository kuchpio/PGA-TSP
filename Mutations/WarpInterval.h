#ifndef __WARP_INTERVAL_MUTATION_H__
#define __WARP_INTERVAL_MUTATION_H__

#include <cuda_runtime.h>
#include <curand_kernel.h>

#define WARP_SIZE 32
#define HALF_WARP_SIZE 16
#define FULL_MASK 0xffffffff
#define LOWER_HALF_MASK 0x0000ffff

namespace tsp{
    
	template <typename gene>
	__device__ __forceinline__ 
	void warpIntervalMutate(gene* chromosome, unsigned int size, curandState* state)
	{
		unsigned int lid = threadIdx.x & (WARP_SIZE - 1);
		unsigned int start, end, geneIndex;
		gene curr;
		bool readNext = true;

		if (lid == 0) {
			start = curand(state) % size;
			end = curand(state) % size;
			if (start > end) 
			{
				unsigned int temp = start;
				start = end;
				end = temp;
			}
		}
		start = __shfl_sync(FULL_MASK, start, 0);
		end = __shfl_sync(FULL_MASK, end, 0) + 1;

		if (lid < HALF_WARP_SIZE) {
			geneIndex = (start & ~(HALF_WARP_SIZE - 1)) + lid;
		} else {
			geneIndex = ((end - 1) & ~(HALF_WARP_SIZE - 1)) + lid - HALF_WARP_SIZE;
		}

		if ((start & ~(HALF_WARP_SIZE - 1)) == ((end - 1) & ~(HALF_WARP_SIZE - 1))) {
			if (lid < HALF_WARP_SIZE) {
				curr = chromosome[geneIndex];
				unsigned int s = start & (HALF_WARP_SIZE - 1);
				unsigned int e = end & (HALF_WARP_SIZE - 1);
				if (e == 0) e = HALF_WARP_SIZE;
				unsigned int shflLid = lid >= s && lid < e ? s + e - 1 - lid : lid; 
				chromosome[geneIndex] = __shfl_sync(LOWER_HALF_MASK, curr, shflLid);
			}
			return;
		}

		while (start < end) {
			if (readNext) {
				curr = chromosome[geneIndex];
				readNext = false;
			}

			unsigned int s = start & (HALF_WARP_SIZE - 1);
			unsigned int e = (end & (HALF_WARP_SIZE - 1)) + HALF_WARP_SIZE;
			if (e == HALF_WARP_SIZE) e = WARP_SIZE;
			unsigned int d = (HALF_WARP_SIZE - s) < (e - HALF_WARP_SIZE) ? (HALF_WARP_SIZE - s) : (e - HALF_WARP_SIZE);

			unsigned int shflLid = (lid >= s && lid < s + d) || (lid < e && lid >= e - d) ? s + e - 1 - lid : lid;
			curr = __shfl_sync(FULL_MASK, curr, shflLid);

			if (lid < HALF_WARP_SIZE && s + d == HALF_WARP_SIZE) {
				chromosome[geneIndex] = curr;
				geneIndex += HALF_WARP_SIZE;
				readNext = true;
			}
			if (lid >= HALF_WARP_SIZE && e - d == HALF_WARP_SIZE) {
				chromosome[geneIndex] = curr;
				geneIndex -= HALF_WARP_SIZE;
				readNext = true;
			}
			start += d;
			end -= d;
		}
	}

}

#endif
