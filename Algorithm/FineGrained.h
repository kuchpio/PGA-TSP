#ifndef __ALGORITHM_FINE_GRAINED_H__
#define __ALGORITHM_FINE_GRAINED_H__

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>

#define WARP_SIZE 32
#define FULL_MASK 0xffffffff

namespace tsp {

	struct IslandGeneticAlgorithmOptions {
		unsigned int islandCount;
		unsigned int islandPopulationSize;
		unsigned int isolatedIterationCount;
		unsigned int migrationCount;
		float crossoverProbability;
		float mutationProbability;
		bool elitism;
	};

	template <typename vertex>
	__device__ __forceinline__ void initializeCycle(vertex* cycle, unsigned int n, curandState *state) {
		vertex lid = (blockDim.x * blockIdx.x + threadIdx.x) & (WARP_SIZE - 1);

		for (vertex i = lid; i < n; i += WARP_SIZE) 
			cycle[i] = i;

		// TODO: Parallelize shuffle
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
	__device__ __forceinline__ int calculateCycleWeight(const vertex* cycle, const Instance instance) {
		unsigned int lid = (blockDim.x * blockIdx.x + threadIdx.x) & (WARP_SIZE - 1);
		unsigned int lidShfl = (lid + 1) & (WARP_SIZE - 1);
		unsigned int n = size(instance);
		unsigned int nWarpSizeAligned = (n & ~(WARP_SIZE - 1)) + WARP_SIZE;

		int sum = 0;
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

	__device__ __forceinline__ int findMin(int current, int index) {
		int currentShuf, indexShuf;
		for (int i = 1; i < WARP_SIZE; i *= 2) {
			currentShuf = __shfl_xor_sync(FULL_MASK, current, i);
			indexShuf = __shfl_xor_sync(FULL_MASK, index, i);
			if (current > currentShuf) {
				index = indexShuf;
				current = currentShuf;
			}
		}
		return index;
	}

	__device__ __forceinline__ int findMax(int current, int index) {
		int currentShuf, indexShuf;
		for (int i = 1; i < WARP_SIZE; i *= 2) {
			currentShuf = __shfl_xor_sync(FULL_MASK, current, i);
			indexShuf = __shfl_xor_sync(FULL_MASK, index, i);
			if (current < currentShuf) {
				index = indexShuf;
				current = currentShuf;
			}
		}
		return index;
	}

	__global__ void setupCurand(curandState* globalState, int seed) {
		int tid = blockIdx.x * blockDim.x + threadIdx.x;
		curand_init(seed, tid, 0, globalState + tid);
	}

	template <typename Instance, typename gene>
	__global__ void initializationKernel(const Instance instance, curandState* globalState, gene* population, int *cycleWeight, int *islandBest, int *islandWorst) {
		__shared__ int s_cycleWeight[WARP_SIZE];
		unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;	// Global thread id
		unsigned int wid = tid >> 5;								// Global warp id
		unsigned int blockWid = wid & (WARP_SIZE - 1);				// Block warp id
		unsigned int baseWid = wid & ~(WARP_SIZE - 1);				// Block base warp id
		unsigned int lid = tid & (WARP_SIZE - 1);					// Warp thread id
		unsigned int n = size(instance);
		unsigned int nWarpSizeAligned = (n & ~(WARP_SIZE - 1)) + WARP_SIZE;
		gene *chromosome = population + wid * nWarpSizeAligned;

		initializeCycle(chromosome, n, globalState + tid);

		__syncthreads();

		int thisCycleWeight = calculateCycleWeight(chromosome, instance);
		if (lid == 0) s_cycleWeight[blockWid] = thisCycleWeight;

		__syncthreads();

		if (blockWid == 0) {
			int value = s_cycleWeight[lid];
			int bestLid = findMin(value, lid);
			int worstLid = findMax(value, lid);
			if (lid == 0) {
				if (bestLid == worstLid) {
					islandBest[blockIdx.x] = 0;
					islandWorst[blockIdx.x] = WARP_SIZE - 1;
				} else {
					islandBest[blockIdx.x] = bestLid;
					islandWorst[blockIdx.x] = worstLid;
				}
			}
		}
		if (blockWid == 1) cycleWeight[baseWid + lid] = s_cycleWeight[lid];
	}

	template <typename gene>
	__global__ void migrationKernel(gene* population, int nWarpSizeAligned, int *cycleWeight, int* islandBest, int* islandWorst) {
		__shared__ int s_widSrcDst[2];
		if (threadIdx.x < WARP_SIZE) {
			int islandPopulationSize = blockDim.x / WARP_SIZE;
			int thisIsland = blockIdx.x;
			int nextIsland = (blockIdx.x + 1) % gridDim.x;
			int nextIslandBestWidGlobal = nextIsland * islandPopulationSize + islandBest[nextIsland];
			int thisIslandWorstWidGlobal = thisIsland * islandPopulationSize + islandWorst[thisIsland];

			// Correct fitness
			int nextIslandBestCycleWeight = cycleWeight[nextIslandBestWidGlobal];
			if (threadIdx.x == 0)
				cycleWeight[thisIslandWorstWidGlobal] = nextIslandBestCycleWeight;

			__syncwarp();

			int thisCycleWeight = cycleWeight[thisIsland * islandPopulationSize + threadIdx.x];

			// Correct islandBest
			if (threadIdx.x == islandBest[thisIsland] && thisCycleWeight > nextIslandBestCycleWeight)
				islandBest[thisIsland] = thisIslandWorstWidGlobal % islandPopulationSize;

			// Correct islandWorst
			int thisIslandWorstWidUpdated = findMax(thisCycleWeight, threadIdx.x);

			if (threadIdx.x == 0) {
				s_widSrcDst[0] = nextIslandBestWidGlobal;
				s_widSrcDst[1] = thisIslandWorstWidGlobal;
				islandWorst[thisIsland] = thisIslandWorstWidUpdated;
			}
		}

		__syncthreads();

		// Replace worst chromosome in thisIsland with best chromosome from nextIsland
		gene* srcChromosome = population + s_widSrcDst[0] * nWarpSizeAligned;
		gene* dstChromosome = population + s_widSrcDst[1] * nWarpSizeAligned;

		for (unsigned int i = threadIdx.x; i < nWarpSizeAligned; i += blockDim.x)
			dstChromosome[i] = srcChromosome[i];
	}

	template <typename Instance, typename gene>
	__global__ void islandEvolutionKernel(const Instance instance, curandState* globalState, unsigned int iterationCount, gene* population, int* cycleWeight, int *islandBest, int *islandWorst) {
		__shared__ int s_cycleWeight[WARP_SIZE];
		__shared__ float s_roulletteWheelThreshold[WARP_SIZE];
		unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;	// Global thread id
		unsigned int wid = tid >> 5;								// Global warp id
		unsigned int blockWid = wid & (WARP_SIZE - 1);				// Block warp id
		unsigned int baseWid = wid & ~(WARP_SIZE - 1);				// Block base warp id
		unsigned int lid = tid & (WARP_SIZE - 1);					// Warp thread id
		unsigned int nWarpSizeAligned = (size(instance) & ~(WARP_SIZE - 1)) + WARP_SIZE;
		gene *chromosome = population + wid * nWarpSizeAligned;

		if (blockWid == 0) s_cycleWeight[lid] = cycleWeight[baseWid + lid];

		int islandBestWid, islandWorstWid;
		if (blockWid == 0) {
			islandBestWid = islandBest[blockIdx.x];
			islandWorstWid = islandWorst[blockIdx.x];
		}

		__syncthreads();

		while (iterationCount-- > 0) {

			// Selection
			{
				if (blockWid == 0) {
					// t(f) = (max - f) / (max - min)
					int min = s_cycleWeight[islandBestWid];
					int max = s_cycleWeight[islandWorstWid];
					float intervalWidth = min == max ? 1.0f : ((float)(max - s_cycleWeight[lid])) / ((float)(max - min));
					float intervalWidthShfl;

					// Warp scan to get prefix sums
					for (unsigned int i = 1; i < WARP_SIZE; i *= 2) {
						intervalWidthShfl = __shfl_up_sync(FULL_MASK, intervalWidth, 1);
						if (lid >= i) intervalWidth += intervalWidthShfl;
					}

					// Save in shared memory
					s_roulletteWheelThreshold[lid] = intervalWidth;
				}

				__syncthreads();

				// Select lane id based on roullette wheel selection
				float threshold = s_roulletteWheelThreshold[lid], rnd;
				if (lid == WARP_SIZE - 1) rnd = curand_uniform(globalState + tid) * threshold;
				rnd = __shfl_sync(FULL_MASK, rnd, WARP_SIZE - 1);
				int selectedWid = WARP_SIZE - __clz(~(__ballot_sync(FULL_MASK, rnd <= threshold)));
				gene* selectedChromosome = population + (baseWid + selectedWid) * nWarpSizeAligned;
				for (unsigned int i = lid; i < nWarpSizeAligned; i += WARP_SIZE) {
					gene buffer = selectedChromosome[i];
					__syncthreads();
					chromosome[i] = buffer;
				}
			}

			__syncthreads();

			// Crossover

			// Mutation

			// Fitness
			{
				int thisCycleWeight = calculateCycleWeight(chromosome, instance);
				if (lid == 0) s_cycleWeight[blockWid] = thisCycleWeight;
			}

			__syncthreads();

			// Best and Worst
			if (blockWid == 0) {
				int thisCycleWeight = s_cycleWeight[lid];
				islandBestWid = findMin(thisCycleWeight, lid);
				islandWorstWid = findMax(thisCycleWeight, lid);
			}
		}

		if (blockWid == 0) {
			cycleWeight[baseWid + lid] = s_cycleWeight[lid];
			if (lid == 0) {
				if (islandBestWid == islandWorstWid) {
					islandBest[blockIdx.x] = 0;
					islandWorst[blockIdx.x] = WARP_SIZE - 1;
				} else {
					islandBest[blockIdx.x] = islandBestWid;
					islandWorst[blockIdx.x] = islandWorstWid;
				}
			}
		}
	}

	template <typename Instance, typename gene = unsigned short>
	int solveTSPFineGrained(const Instance instance, struct IslandGeneticAlgorithmOptions options, int seed) {
		unsigned int n = size(instance);
		unsigned int nWarpSizeAligned = (n & ~(WARP_SIZE - 1)) + WARP_SIZE;
		int *d_cycleWeight, *d_islandBest, *d_islandWorst;
		gene* d_population;
		curandState* d_globalState;
		int* h_cycleWeight = new int[options.islandCount * options.islandPopulationSize];
		int* h_islandBest = new int[options.islandCount];
		int* h_islandWorst = new int[options.islandCount];

		if (cudaMalloc(&d_cycleWeight, options.islandCount * options.islandPopulationSize * sizeof(int)) != cudaSuccess)
			return -1;

		if (cudaMalloc(&d_population, nWarpSizeAligned * options.islandCount * options.islandPopulationSize * sizeof(gene)) != cudaSuccess)
			return -1;

		if (cudaMalloc(&d_globalState, options.islandCount * options.islandPopulationSize * WARP_SIZE * sizeof(curandState)) != cudaSuccess)
			return -1;

		if (cudaMalloc(&d_islandBest, options.islandCount * sizeof(int)) != cudaSuccess)
			return -1;

		if (cudaMalloc(&d_islandWorst, options.islandCount * sizeof(int)) != cudaSuccess)
			return -1;

		setupCurand<<<options.islandCount, options.islandPopulationSize * WARP_SIZE>>>(d_globalState, seed);

		initializationKernel<<<options.islandCount, options.islandPopulationSize * WARP_SIZE>>>(instance, d_globalState, d_population, d_cycleWeight, d_islandBest, d_islandWorst);

		if (cudaDeviceSynchronize() != cudaSuccess)
			return -1;

		if (cudaMemcpy(h_cycleWeight, d_cycleWeight, options.islandCount * options.islandPopulationSize * sizeof(int), cudaMemcpyDeviceToHost) != cudaSuccess)
			return -1;

		if (cudaMemcpy(h_islandBest, d_islandBest, options.islandCount * sizeof(int), cudaMemcpyDeviceToHost) != cudaSuccess)
			return -1;

		if (cudaMemcpy(h_islandWorst, d_islandWorst, options.islandCount * sizeof(int), cudaMemcpyDeviceToHost) != cudaSuccess)
			return -1;

		std::cout << "Island:\t";
		for (unsigned int i = 0; i < options.islandCount; i++)
			std::cout << i << "\t\t";
		std::cout << "\nINITIAL\nBest:\t";
		for (unsigned int i = 0; i < options.islandCount; i++)
			std::cout << h_cycleWeight[i * options.islandPopulationSize + h_islandBest[i]] << "(" << h_islandBest[i] << ")\t";
		std::cout << "\nWorst:\t";
		for (unsigned int i = 0; i < options.islandCount; i++)
			std::cout << h_cycleWeight[i * options.islandPopulationSize + h_islandWorst[i]] << "(" << h_islandWorst[i] << ")\t";

		for (unsigned int migrationNumber = 1; migrationNumber <= options.migrationCount; migrationNumber++) {

			migrationKernel<<<options.islandCount, options.islandPopulationSize * WARP_SIZE>>>(d_population, nWarpSizeAligned, d_cycleWeight, d_islandBest, d_islandWorst);

			islandEvolutionKernel<<<options.islandCount, options.islandPopulationSize * WARP_SIZE>>>(instance, d_globalState, options.isolatedIterationCount, d_population, d_cycleWeight, d_islandBest, d_islandWorst);
			
			if (cudaDeviceSynchronize() != cudaSuccess)
				return -1;

			if (cudaMemcpy(h_cycleWeight, d_cycleWeight, options.islandCount * options.islandPopulationSize * sizeof(int), cudaMemcpyDeviceToHost) != cudaSuccess)
				return -1;

			if (cudaMemcpy(h_islandBest, d_islandBest, options.islandCount * sizeof(int), cudaMemcpyDeviceToHost) != cudaSuccess)
				return -1;

			if (cudaMemcpy(h_islandWorst, d_islandWorst, options.islandCount * sizeof(int), cudaMemcpyDeviceToHost) != cudaSuccess)
				return -1;

			std::cout << "\nCYCLE: " << migrationNumber << "\nBest:\t";
			for (unsigned int i = 0; i < options.islandCount; i++)
				std::cout << h_cycleWeight[i * options.islandPopulationSize + h_islandBest[i]] << "(" << h_islandBest[i] << ")\t";
			std::cout << "\nWorst:\t";
			for (unsigned int i = 0; i < options.islandCount; i++)
				std::cout << h_cycleWeight[i * options.islandPopulationSize + h_islandWorst[i]] << "(" << h_islandWorst[i] << ")\t";
		}

		cudaFree(d_cycleWeight);
		cudaFree(d_population);
		cudaFree(d_globalState);
		cudaFree(d_islandBest);
		cudaFree(d_islandWorst);

		delete[] h_cycleWeight;
		delete[] h_islandBest;
		delete[] h_islandWorst;
		
		return 0;
	}

}

#endif __ALGORITHM_FINE_GRAINED_H__
