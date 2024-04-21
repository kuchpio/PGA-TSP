#ifndef __ALGORITHM_FINE_GRAINED_H__
#define __ALGORITHM_FINE_GRAINED_H__

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>

#define WARP_SIZE 32
#define FULL_MASK 0xffffffff

namespace tsp {

	__device__ __forceinline__ void initializeChromosome(int* chromosome, unsigned int n, curandState *state) {
		unsigned int lid = (blockDim.x * blockIdx.x + threadIdx.x) & (WARP_SIZE - 1);

		for (unsigned int i = 0; i < n; i += WARP_SIZE) {
			chromosome[i + lid] = i + lid;
		}

		// TODO: Parallelize shuffle
		if (lid == 0) {
			for (unsigned int i = n - 1; i > 0; i--) {
				unsigned int j = curand(state) % (i + 1);

				// Swap chromosome[i] with chromosome[j]
				int temp = chromosome[i];
				chromosome[i] = chromosome[j];
				chromosome[j] = temp;
			}
		}
	}

	template <typename Instance>
	__device__ __forceinline__ void calculateFitness(const int* chromosome, const Instance instance, int* fitness) {
		unsigned int lid = (blockDim.x * blockIdx.x + threadIdx.x) & (WARP_SIZE - 1);
		unsigned int lidShfl = (lid + 1) & (WARP_SIZE - 1);
		unsigned int n = size(instance);
		unsigned int nWarpSizeAligned = (n & ~(WARP_SIZE - 1)) + WARP_SIZE;

		int sum = 0;
		int from = chromosome[lid];
		int to = __shfl_sync(FULL_MASK, from, lidShfl);
		int first, last;

		if (lid < WARP_SIZE - 1) {
			sum += edgeWeight(instance, from, to);
		} else {
			first = to;
			last = from;
		}

		for (unsigned int i = WARP_SIZE; i < nWarpSizeAligned - WARP_SIZE; i += WARP_SIZE) {
			from = chromosome[i + lid];
			to = __shfl_sync(FULL_MASK, from, lidShfl);
			sum += edgeWeight(instance, lid == WARP_SIZE - 1 ? last : from, to);
			if (lid == WARP_SIZE - 1) last = from;
		}

		from = chromosome[nWarpSizeAligned - WARP_SIZE + lid];
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

		if (lid == 0) *fitness = sum;
	}

	__device__ __forceinline__ int findBestFitness(int current, int index) {
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

	__device__ __forceinline__ int findWorstFitness(int current, int index) {
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

	template <typename Instance>
	__global__ void initializationKernel(const Instance instance, curandState* globalState, int* population, int *fitness, int *islandBest, int *islandWorst) {
		__shared__ int s_fitness[WARP_SIZE];
		unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;	// Global thread id
		unsigned int wid = tid >> 5;								// Global warp id
		unsigned int blockWid = wid & (WARP_SIZE - 1);				// Block warp id
		unsigned int baseWid = wid & ~(WARP_SIZE - 1);				// Block base warp id
		unsigned int lid = tid & (WARP_SIZE - 1);					// Warp thread id
		unsigned int n = size(instance);
		unsigned int nWarpSizeAligned = (n & ~(WARP_SIZE - 1)) + WARP_SIZE;
		int *chromosome = population + wid * nWarpSizeAligned;

		initializeChromosome(chromosome, n, globalState + tid);

		__syncthreads();

		calculateFitness(chromosome, instance, s_fitness + blockWid);

		__syncthreads();

		if (blockWid == 0) {
			int value = s_fitness[lid];
			int bestLid = findBestFitness(value, lid);
			int worstLid = findWorstFitness(value, lid);
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
		if (blockWid == 1) fitness[baseWid + lid] = s_fitness[lid];
	}

	__global__ void migrationKernel(int* population, int nWarpSizeAligned, int *fitness, int* islandBest, int* islandWorst) {
		__shared__ int s_widSrcDst[2];
		if (threadIdx.x < WARP_SIZE) {
			int islandPopulationSize = blockDim.x / WARP_SIZE;
			int thisIsland = blockIdx.x;
			int nextIsland = (blockIdx.x + 1) % gridDim.x;
			int nextIslandBestWidGlobal = nextIsland * islandPopulationSize + islandBest[nextIsland];
			int thisIslandWorstWidGlobal = thisIsland * islandPopulationSize + islandWorst[thisIsland];

			// Correct fitness
			int nextIslandBestFitness = fitness[nextIslandBestWidGlobal];
			if (threadIdx.x == 0)
				fitness[thisIslandWorstWidGlobal] = nextIslandBestFitness;

			__syncwarp();

			int thisFitness = fitness[thisIsland * islandPopulationSize + threadIdx.x];

			// Correct islandBest
			if (threadIdx.x == islandBest[thisIsland] && thisFitness > nextIslandBestFitness)
				islandBest[thisIsland] = thisIslandWorstWidGlobal % islandPopulationSize;

			// Correct islandWorst
			int thisIslandWorstWidUpdated = findWorstFitness(thisFitness, threadIdx.x);

			if (threadIdx.x == 0) {
				s_widSrcDst[0] = nextIslandBestWidGlobal;
				s_widSrcDst[1] = thisIslandWorstWidGlobal;
				islandWorst[thisIsland] = thisIslandWorstWidUpdated;
			}
		}

		__syncthreads();

		// Replace worst chromosome in thisIsland with best chromosome from nextIsland
		int* srcChromosome = population + s_widSrcDst[0] * nWarpSizeAligned;
		int* dstChromosome = population + s_widSrcDst[1] * nWarpSizeAligned;

		for (unsigned int i = threadIdx.x; i < nWarpSizeAligned; i += blockDim.x)
			dstChromosome[i] = srcChromosome[i];
	}

	template <typename Instance>
	__global__ void islandEvolutionKernel(const Instance instance, curandState* globalState, unsigned int iterationCount, int* population, int* fitness, int *islandBest, int *islandWorst) {
		__shared__ int s_fitness[WARP_SIZE];
		__shared__ float s_roulletteWheelThreshold[WARP_SIZE];
		unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;	// Global thread id
		unsigned int wid = tid >> 5;								// Global warp id
		unsigned int blockWid = wid & (WARP_SIZE - 1);				// Block warp id
		unsigned int baseWid = wid & ~(WARP_SIZE - 1);				// Block base warp id
		unsigned int lid = tid & (WARP_SIZE - 1);					// Warp thread id
		unsigned int nWarpSizeAligned = (size(instance) & ~(WARP_SIZE - 1)) + WARP_SIZE;
		int *chromosome = population + wid * nWarpSizeAligned;

		if (blockWid == 0) s_fitness[lid] = fitness[baseWid + lid];

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
					int min = s_fitness[islandBestWid];
					int max = s_fitness[islandWorstWid];
					float intervalWidth = min == max ? 1.0f : ((float)(max - s_fitness[lid])) / ((float)(max - min));
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
				int* selectedChromosome = population + (baseWid + selectedWid) * nWarpSizeAligned;
				for (unsigned int i = lid; i < nWarpSizeAligned; i += WARP_SIZE) {
					int buffer = selectedChromosome[i];
					__syncthreads();
					chromosome[i] = buffer;
				}
			}

			__syncthreads();

			// Crossover

			// Mutation

			// Fitness
			calculateFitness(chromosome, instance, s_fitness + blockWid);

			__syncthreads();

			// Best and Worst
			if (blockWid == 0) {
				int value = s_fitness[lid];
				islandBestWid = findBestFitness(value, lid);
				islandWorstWid = findWorstFitness(value, lid);
			}
		}

		if (blockWid == 0) {
			fitness[baseWid + lid] = s_fitness[lid];
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

	template <typename Instance>
	int solveTSPFineGrained(const Instance instance, unsigned int islandCount, unsigned int isolatedIterationCount, unsigned int migrationCount, int seed) {
		unsigned int islandPopulationSize = 1024 / WARP_SIZE;
		unsigned int n = size(instance);
		unsigned int nWarpSizeAligned = (n & ~(WARP_SIZE - 1)) + WARP_SIZE;
		int *d_fitness, *d_population, *d_islandBest, *d_islandWorst;
		curandState* d_globalState;
		int* h_fitness = new int[islandCount * islandPopulationSize];
		int* h_islandBest = new int[islandCount];
		int* h_islandWorst = new int[islandCount];

		if (cudaMalloc(&d_fitness, islandCount * islandPopulationSize * sizeof(int)) != cudaSuccess)
			return -1;

		if (cudaMalloc(&d_population, nWarpSizeAligned * islandCount * islandPopulationSize * sizeof(int)) != cudaSuccess)
			return -1;

		if (cudaMalloc(&d_globalState, islandCount * islandPopulationSize * WARP_SIZE * sizeof(curandState)) != cudaSuccess)
			return -1;

		if (cudaMalloc(&d_islandBest, islandCount * sizeof(int)) != cudaSuccess)
			return -1;

		if (cudaMalloc(&d_islandWorst, islandCount * sizeof(int)) != cudaSuccess)
			return -1;

		setupCurand<<<islandCount, islandPopulationSize* WARP_SIZE>>>(d_globalState, seed);

		initializationKernel<<<islandCount, islandPopulationSize * WARP_SIZE>>>(instance, d_globalState, d_population, d_fitness, d_islandBest, d_islandWorst);

		if (cudaDeviceSynchronize() != cudaSuccess)
			return -1;

		if (cudaMemcpy(h_fitness, d_fitness, islandCount * islandPopulationSize * sizeof(int), cudaMemcpyDeviceToHost) != cudaSuccess)
			return -1;

		if (cudaMemcpy(h_islandBest, d_islandBest, islandCount * sizeof(int), cudaMemcpyDeviceToHost) != cudaSuccess)
			return -1;

		if (cudaMemcpy(h_islandWorst, d_islandWorst, islandCount * sizeof(int), cudaMemcpyDeviceToHost) != cudaSuccess)
			return -1;

		std::cout << "Island:\t";
		for (unsigned int i = 0; i < islandCount; i++)
			std::cout << i << "\t\t";
		std::cout << "\nINITIAL\nBest:\t";
		for (unsigned int i = 0; i < islandCount; i++)
			std::cout << h_fitness[i * islandPopulationSize + h_islandBest[i]] << "(" << h_islandBest[i] << ")\t";
		std::cout << "\nWorst:\t";
		for (unsigned int i = 0; i < islandCount; i++)
			std::cout << h_fitness[i * islandPopulationSize + h_islandWorst[i]] << "(" << h_islandWorst[i] << ")\t";

		for (unsigned int migrationNumber = 1; migrationNumber <= migrationCount; migrationNumber++) {

			migrationKernel<<<islandCount, islandPopulationSize * WARP_SIZE>>>(d_population, nWarpSizeAligned, d_fitness, d_islandBest, d_islandWorst);

			islandEvolutionKernel<<<islandCount, islandPopulationSize * WARP_SIZE>>>(instance, d_globalState, isolatedIterationCount, d_population, d_fitness, d_islandBest, d_islandWorst);
			
			if (cudaDeviceSynchronize() != cudaSuccess)
				return -1;

			if (cudaMemcpy(h_fitness, d_fitness, islandCount * islandPopulationSize * sizeof(int), cudaMemcpyDeviceToHost) != cudaSuccess)
				return -1;

			if (cudaMemcpy(h_islandBest, d_islandBest, islandCount * sizeof(int), cudaMemcpyDeviceToHost) != cudaSuccess)
				return -1;

			if (cudaMemcpy(h_islandWorst, d_islandWorst, islandCount * sizeof(int), cudaMemcpyDeviceToHost) != cudaSuccess)
				return -1;

			std::cout << "\nCYCLE: " << migrationNumber << "\nBest:\t";
			for (unsigned int i = 0; i < islandCount; i++)
				std::cout << h_fitness[i * islandPopulationSize + h_islandBest[i]] << "(" << h_islandBest[i] << ")\t";
			std::cout << "\nWorst:\t";
			for (unsigned int i = 0; i < islandCount; i++)
				std::cout << h_fitness[i * islandPopulationSize + h_islandWorst[i]] << "(" << h_islandWorst[i] << ")\t";
		}

		cudaFree(d_fitness);
		cudaFree(d_population);
		cudaFree(d_globalState);
		cudaFree(d_islandBest);
		cudaFree(d_islandWorst);

		delete[] h_fitness;
		delete[] h_islandBest;
		delete[] h_islandWorst;
		
		return 0;
	}

}

#endif __ALGORITHM_FINE_GRAINED_H__
