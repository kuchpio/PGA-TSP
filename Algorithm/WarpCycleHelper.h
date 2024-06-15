#ifndef __WARP_CYCLE_HELPER_H__
#define __WARP_CYCLE_HELPER_H__

#define WARP_SIZE 32
#define FULL_MASK 0xffffffff

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>

namespace tsp {

	template <typename vertex>
	__device__ __forceinline__ void initializeCycle(vertex* cycle, unsigned int n, curandState *state) 
	{
		vertex lid = (blockDim.x * blockIdx.x + threadIdx.x) & (WARP_SIZE - 1);

		for (vertex i = lid; i < n; i += WARP_SIZE) 
			cycle[i] = i;

		if (lid == 0) {
			for (unsigned int i = n - 1; i > 0; i--) {
				unsigned int j = curand(state) % (i + 1);

				// Swap chromosome[i] with chromosome[j]
				vertex temp = cycle[i];
				cycle[i] = cycle[j];
				cycle[j] = temp;
			}
		}
	}

	template <typename Instance, typename vertex>
	__device__ __forceinline__ unsigned int calculateCycleWeight(const vertex* cycle, const Instance instance) 
	{
		unsigned int lid = threadIdx.x & (WARP_SIZE - 1);
		unsigned int lidShfl = (lid + 1) & (WARP_SIZE - 1);
		unsigned int n = size(instance);
		unsigned int nWarpSizeAligned = (n & ~(WARP_SIZE - 1)) + WARP_SIZE;

		unsigned int sum = 0;
		vertex from = cycle[lid];
		vertex to = __shfl_sync(FULL_MASK, from, lidShfl);
		vertex first, last;

		if (lid < WARP_SIZE - 1) {
			sum += edgeWeight(instance, from, to);
		} else {
			first = to;
			last = from;
		}

		for (unsigned int i = WARP_SIZE; i < nWarpSizeAligned - WARP_SIZE; i += WARP_SIZE) {
			from = cycle[i + lid];
			to = __shfl_sync(FULL_MASK, from, lidShfl);
			sum += edgeWeight(instance, lid == WARP_SIZE - 1 ? last : from, to);
			if (lid == WARP_SIZE - 1) last = from;
		}

		from = cycle[nWarpSizeAligned - WARP_SIZE + lid];
		to = __shfl_sync(FULL_MASK, from, lidShfl);
		if (nWarpSizeAligned - WARP_SIZE + lid < n - 1)
			sum += edgeWeight(instance, from, to);

		if (lid == WARP_SIZE - 1)
			sum += edgeWeight(instance, last, to);

		last = __shfl_sync(FULL_MASK, from, (n - 1) & (WARP_SIZE - 1));

		if (lid == WARP_SIZE - 1)
			sum += edgeWeight(instance, last, first);

		for (unsigned int i = 1; i < WARP_SIZE; i *= 2) 
			sum += __shfl_xor_sync(FULL_MASK, sum, i);

		return sum;
	}

}

#endif
