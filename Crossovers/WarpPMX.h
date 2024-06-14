#ifndef __INTERVAL_CROSSOVER_H__
#define __INTERVAL_CROSSOVER_H__

#define WARP_SIZE 32
#define FULL_MASK 0xffffffff

#include <cuda_runtime.h>
#include <curand_kernel.h>

namespace tsp {

	// Interval swapped using warp reads and writes
	// Fixes are done using only warp leader
	template<typename gene>
	__device__ __forceinline__
	void crossover1(gene* chromosomeA, gene* chromosomeB, unsigned int size, curandState* state)
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

			while (indexOffset + WARP_SIZE < end) {
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

		if (lid == 0) {
			for (unsigned int indexOffset = 0; indexOffset < start; indexOffset++) {
				gene a = chromosomeA[indexOffset];
				gene b = chromosomeB[indexOffset];
				bool aUnique = false, bUnique = false;

				while (!(aUnique && bUnique)) {
					aUnique = bUnique = true;

					for (unsigned int i = start; i < end; i++) {
						gene aInterval = chromosomeA[i];
						gene bInterval = chromosomeB[i];

						if (a == aInterval) {
							aUnique = false;
							a = bInterval;
						}
						if (b == bInterval) {
							bUnique = false;
							b = aInterval;
						}
					}
				}
				chromosomeA[indexOffset] = a;
				chromosomeB[indexOffset] = b;
			}

			for (unsigned int indexOffset = end; indexOffset < size; indexOffset++) {
				gene a = chromosomeA[indexOffset];
				gene b = chromosomeB[indexOffset];
				bool aUnique = false, bUnique = false;

				while (!(aUnique && bUnique)) {
					aUnique = bUnique = true;

					for (unsigned int i = start; i < end; i++) {
						gene aInterval = chromosomeA[i];
						gene bInterval = chromosomeB[i];

						if (a == aInterval) {
							aUnique = false;
							a = bInterval;
						}
						if (b == bInterval) {
							bUnique = false;
							b = aInterval;
						}
					}
				}
				chromosomeA[indexOffset] = a;
				chromosomeB[indexOffset] = b;
			}
		}
	}

	// Interval swapped using warp reads and writes
	// Outside of interval is read and written unsing warp
	// Uniqeness is checked using warp broadcast sequential reads of swapped interval
	template<typename gene>
	__device__ __forceinline__
	void crossover2(gene* chromosomeA, gene* chromosomeB, unsigned int size, curandState* state)
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

			while (indexOffset + WARP_SIZE < end) {
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

			while (__any_sync(FULL_MASK, !(aUnique && bUnique))) {
				aUnique = bUnique = true;

				for (unsigned int i = start; i < end; i++) {
					gene aInterval = chromosomeA[i];
					gene bInterval = chromosomeB[i];

					if (a == aInterval && indexOffset + lid < start) {
						aUnique = false;
						a = bInterval;
					}
					if (b == bInterval && indexOffset + lid < start) {
						bUnique = false;
						b = aInterval;
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

			while (__any_sync(FULL_MASK, !(aUnique && bUnique))) {
				aUnique = bUnique = true;

				for (unsigned int i = start; i < end; i++) {
					gene aInterval = chromosomeA[i];
					gene bInterval = chromosomeB[i];

					if (a == aInterval && end <= indexOffset + lid && indexOffset + lid < size) {
						aUnique = false;
						a = bInterval;
					}
					if (b == bInterval && end <= indexOffset + lid && indexOffset + lid < size) {
						bUnique = false;
						b = aInterval;
					}
				}
			}

			chromosomeA[indexOffset + lid] = a;
			chromosomeB[indexOffset + lid] = b;
		}

	}

	// Interval swapped using warp reads and writes
	// Outside of interval is read and written unsing warp
	// Uniqeness is checked using sequential __shfl_sync's of coalesced reads of swapped interval
	template<typename gene>
	__device__ __forceinline__
	void crossover3(gene* chromosomeA, gene* chromosomeB, unsigned int size, curandState* state)
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

			while (indexOffset + WARP_SIZE < end) {
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

			while (__any_sync(FULL_MASK, !(aUnique && bUnique))) {
				aUnique = bUnique = true;

				for (unsigned int intervalIndexOffset = start & ~(WARP_SIZE - 1); intervalIndexOffset < end; intervalIndexOffset += WARP_SIZE) {
					gene aInterval = chromosomeA[intervalIndexOffset + lid];
					gene bInterval = chromosomeB[intervalIndexOffset + lid];

					unsigned int broadcastStartLid = (intervalIndexOffset > start ? intervalIndexOffset : start) & (WARP_SIZE - 1);
					unsigned int broadcastEndLid = ((intervalIndexOffset + WARP_SIZE < end ? intervalIndexOffset + WARP_SIZE : end) - 1) & (WARP_SIZE - 1);
					for (unsigned int broadcastLid = broadcastStartLid; broadcastLid <= broadcastEndLid; broadcastLid++) {
						gene aIntervalShfl = __shfl_sync(FULL_MASK, aInterval, broadcastLid);
						gene bIntervalShfl = __shfl_sync(FULL_MASK, bInterval, broadcastLid);

						if (a == aIntervalShfl && indexOffset + lid < start) {
							aUnique = false;
							a = bIntervalShfl;
						}
						if (b == bIntervalShfl && indexOffset + lid < start) {
							bUnique = false;
							b = aIntervalShfl;
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

			while (__any_sync(FULL_MASK, !(aUnique && bUnique))) {
				aUnique = bUnique = true;

				for (unsigned int intervalIndexOffset = start & ~(WARP_SIZE - 1); intervalIndexOffset < end; intervalIndexOffset += WARP_SIZE) {
					gene aInterval = chromosomeA[intervalIndexOffset + lid];
					gene bInterval = chromosomeB[intervalIndexOffset + lid];

					unsigned int broadcastStartLid = (intervalIndexOffset > start ? intervalIndexOffset : start) & (WARP_SIZE - 1);
					unsigned int broadcastEndLid = ((intervalIndexOffset + WARP_SIZE < end ? intervalIndexOffset + WARP_SIZE : end) - 1) & (WARP_SIZE - 1);
					for (unsigned int broadcastLid = broadcastStartLid; broadcastLid <= broadcastEndLid; broadcastLid++) {
						gene aIntervalShfl = __shfl_sync(FULL_MASK, aInterval, broadcastLid);
						gene bIntervalShfl = __shfl_sync(FULL_MASK, bInterval, broadcastLid);

						if (a == aIntervalShfl && end <= indexOffset + lid && indexOffset + lid < size) {
							aUnique = false;
							a = bIntervalShfl;
						}
						if (b == bIntervalShfl && end <= indexOffset + lid && indexOffset + lid < size) {
							bUnique = false;
							b = aIntervalShfl;
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
