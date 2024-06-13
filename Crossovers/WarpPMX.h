#ifndef __INTERVAL_CROSSOVER_H__
#define __INTERVAL_CROSSOVER_H__

#define WARP_SIZE 32
#define FULL_MASK 0xffffffff

#include <cuda_runtime.h>
#include <curand_kernel.h>

namespace tsp {

	template<typename gene>
	__device__ __forceinline__
	void crossover(gene* chromosomeA, gene* chromosomeB, unsigned int size, curandState* state)
	{
		unsigned int lid = threadIdx.x & (WARP_SIZE - 1);
		unsigned int start, end;

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

		// Swap interval <start; end)
		{
			unsigned int indexOffset = start & ~(WARP_SIZE - 1);
			gene temp;

			gene a = chromosomeA[indexOffset + lid];
			gene b = chromosomeB[indexOffset + lid];
			if (start <= indexOffset + lid && indexOffset + lid < end) {
				temp = a;
				a = b;
				b = temp;
			}
			chromosomeA[indexOffset + lid] = a;
			chromosomeB[indexOffset + lid] = b;
			indexOffset += WARP_SIZE;

			while (indexOffset < end - WARP_SIZE) {
				temp = chromosomeA[indexOffset + lid];
				chromosomeA[indexOffset + lid] = chromosomeB[indexOffset + lid];
				chromosomeB[indexOffset + lid] = temp;
				indexOffset += WARP_SIZE;
			}

			if (indexOffset < end) {
				a = chromosomeA[indexOffset + lid];
				b = chromosomeB[indexOffset + lid];
				if (start <= indexOffset + lid && indexOffset + lid < end) {
					temp = a;
					a = b;
					b = temp;
				}
				chromosomeA[indexOffset + lid] = a;
				chromosomeB[indexOffset + lid] = b;
			}
		}

		// Fix interval <0; start)
		for (unsigned int indexOffset = 0; indexOffset < start; indexOffset += WARP_SIZE) {
			gene a = chromosomeA[indexOffset + lid];
			gene b = chromosomeB[indexOffset + lid];
			bool aUnique = false, bUnique = false;

			while (__any_sync(FULL_MASK, !aUnique || !bUnique)) {
				aUnique = bUnique = true;
				unsigned int intervalIndexOffset = start & ~(WARP_SIZE - 1);
				gene aInterval = chromosomeA[intervalIndexOffset + lid];
				gene bInterval = chromosomeB[intervalIndexOffset + lid];
				unsigned int broadcastLidEnd = intervalIndexOffset + WARP_SIZE < end ? WARP_SIZE : end & (WARP_SIZE - 1);
				for (unsigned int broadcastLid = start & (WARP_SIZE - 1); broadcastLid < broadcastLidEnd; broadcastLid++) {
					gene aIntervalShuf = __shfl_sync(FULL_MASK, aInterval, broadcastLid);
					gene bIntervalShuf = __shfl_sync(FULL_MASK, bInterval, broadcastLid);
					if (a == aIntervalShuf && indexOffset + lid < start) {
						aUnique = false;
						a = bIntervalShuf;
					}
					if (b == bIntervalShuf && indexOffset + lid < start) {
						bUnique = false;
						b = aIntervalShuf;
					}
				}
				intervalIndexOffset += WARP_SIZE;

				while (intervalIndexOffset < end - WARP_SIZE) {
					aInterval = chromosomeA[intervalIndexOffset + lid];
					bInterval = chromosomeB[intervalIndexOffset + lid];
					for (unsigned int broadcastLid = 0; broadcastLid < WARP_SIZE; broadcastLid++) {
						gene aIntervalShuf = __shfl_sync(FULL_MASK, aInterval, broadcastLid);
						gene bIntervalShuf = __shfl_sync(FULL_MASK, bInterval, broadcastLid);
						if (a == aIntervalShuf && indexOffset + lid < start) {
							aUnique = false;
							a = bIntervalShuf;
						}
						if (b == bIntervalShuf && indexOffset + lid < start) {
							bUnique = false;
							b = aIntervalShuf;
						}
					}
					intervalIndexOffset += WARP_SIZE;
				}

				if (intervalIndexOffset < end) {
					aInterval = chromosomeA[intervalIndexOffset + lid];
					bInterval = chromosomeB[intervalIndexOffset + lid];
					for (unsigned int broadcastLid = 0; broadcastLid < (end & (WARP_SIZE - 1)); broadcastLid++) {
						gene aIntervalShuf = __shfl_sync(FULL_MASK, aInterval, broadcastLid);
						gene bIntervalShuf = __shfl_sync(FULL_MASK, bInterval, broadcastLid);
						if (a == aIntervalShuf && indexOffset + lid < start) {
							aUnique = false;
							a = bIntervalShuf;
						}
						if (b == bIntervalShuf && indexOffset + lid < start) {
							bUnique = false;
							b = aIntervalShuf;
						}
					}
				}
			}

			chromosomeA[indexOffset + lid] = a;
			chromosomeB[indexOffset + lid] = b;
		}

		// Fix interval <end; size)
		for (unsigned int indexOffset = end & ~(WARP_SIZE - 1); indexOffset < size; indexOffset += WARP_SIZE) {
			gene a = chromosomeA[indexOffset + lid];
			gene b = chromosomeB[indexOffset + lid];
			bool aUnique = false, bUnique = false;

			while (__any_sync(FULL_MASK, !aUnique || !bUnique)) {
				aUnique = bUnique = true;
				unsigned int intervalIndexOffset = start & ~(WARP_SIZE - 1);
				gene aInterval = chromosomeA[intervalIndexOffset + lid];
				gene bInterval = chromosomeB[intervalIndexOffset + lid];
				unsigned int broadcastLidEnd = intervalIndexOffset + WARP_SIZE < end ? WARP_SIZE : end & (WARP_SIZE - 1);
				for (unsigned int broadcastLid = start & (WARP_SIZE - 1); broadcastLid < broadcastLidEnd; broadcastLid++) {
					gene aIntervalShuf = __shfl_sync(FULL_MASK, aInterval, broadcastLid);
					gene bIntervalShuf = __shfl_sync(FULL_MASK, bInterval, broadcastLid);
					if (a == aIntervalShuf && end <= indexOffset + lid && indexOffset + lid < size) {
						aUnique = false;
						a = bIntervalShuf;
					}
					if (b == bIntervalShuf && end <= indexOffset + lid && indexOffset + lid < size) {
						bUnique = false;
						b = aIntervalShuf;
					}
				}
				intervalIndexOffset += WARP_SIZE;

				while (intervalIndexOffset < end - WARP_SIZE) {
					aInterval = chromosomeA[intervalIndexOffset + lid];
					bInterval = chromosomeB[intervalIndexOffset + lid];
					for (unsigned int broadcastLid = 0; broadcastLid < WARP_SIZE; broadcastLid++) {
						gene aIntervalShuf = __shfl_sync(FULL_MASK, aInterval, broadcastLid);
						gene bIntervalShuf = __shfl_sync(FULL_MASK, bInterval, broadcastLid);
						if (a == aIntervalShuf && end <= indexOffset + lid && indexOffset + lid < size) {
							aUnique = false;
							a = bIntervalShuf;
						}
						if (b == bIntervalShuf && end <= indexOffset + lid && indexOffset + lid < size) {
							bUnique = false;
							b = aIntervalShuf;
						}
					}
					intervalIndexOffset += WARP_SIZE;
				}

				if (intervalIndexOffset < end) {
					aInterval = chromosomeA[intervalIndexOffset + lid];
					bInterval = chromosomeB[intervalIndexOffset + lid];
					for (unsigned int broadcastLid = 0; broadcastLid < (end & (WARP_SIZE - 1)); broadcastLid++) {
						gene aIntervalShuf = __shfl_sync(FULL_MASK, aInterval, broadcastLid);
						gene bIntervalShuf = __shfl_sync(FULL_MASK, bInterval, broadcastLid);
						if (a == aIntervalShuf && end <= indexOffset + lid && indexOffset + lid < size) {
							aUnique = false;
							a = bIntervalShuf;
						}
						if (b == bIntervalShuf && end <= indexOffset + lid && indexOffset + lid < size) {
							bUnique = false;
							b = aIntervalShuf;
						}
					}
				}
			}

			chromosomeA[indexOffset + lid] = a;
			chromosomeB[indexOffset + lid] = b;
		}

	}

}

#endif
