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
	__device__ __forceinline__ unsigned int calculateCycleWeight(const vertex* cycle, const Instance instance) {
		unsigned int lid = (blockDim.x * blockIdx.x + threadIdx.x) & (WARP_SIZE - 1);
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
	__global__ void initializationKernel(const Instance instance, curandState* globalState, gene* population, unsigned int islandPopulationSize, unsigned int* cycleWeight, unsigned int* islandBest, unsigned int* islandWorst) {
		extern __shared__ unsigned int s_buffer[];
		unsigned int *s_reducedMin = s_buffer;
		unsigned int *s_reducedMinIndex = s_reducedMin + WARP_SIZE;
		unsigned int *s_reducedMax = s_reducedMinIndex + WARP_SIZE;
		unsigned int *s_reducedMaxIndex = s_reducedMax + WARP_SIZE;
		unsigned int *s_cycleWeight = s_buffer + 4 * WARP_SIZE;

		unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;	// Global thread id
		unsigned int blockWid = (tid >> 5) & (WARP_SIZE - 1);		// Block warp id
		unsigned int lid = tid & (WARP_SIZE - 1);					// Warp thread id
		unsigned int n = size(instance);
		unsigned int nWarpSizeAligned = (n & ~(WARP_SIZE - 1)) + WARP_SIZE;

		for (unsigned int chromosomeIndex = blockWid; chromosomeIndex < islandPopulationSize; chromosomeIndex += (blockDim.x >> 5)) {
			gene* chromosome = population + (blockIdx.x * 2 * islandPopulationSize + chromosomeIndex) * nWarpSizeAligned;

			initializeCycle(chromosome, n, globalState + tid);

			unsigned int thisCycleWeight = calculateCycleWeight(chromosome, instance);
			if (lid == 0) s_cycleWeight[chromosomeIndex] = thisCycleWeight;
		}

		__syncthreads();

		// 1. Block stride loop thread reduction and output writing
		unsigned int min = 0xffffffff, minIndex = threadIdx.x, max = 0, maxIndex = threadIdx.x;
		for (unsigned int chromosomeIndex = threadIdx.x; chromosomeIndex < islandPopulationSize; chromosomeIndex += blockDim.x) {

			unsigned int value = s_cycleWeight[chromosomeIndex];
			if (min > value) {
				min = value;
				minIndex = chromosomeIndex;
			}
			if (max < value) {
				max = value;
				maxIndex = chromosomeIndex;
			}

			cycleWeight[blockIdx.x * islandPopulationSize + chromosomeIndex] = value;
		}

		// 2. Warp reduction in each warp
		unsigned int minShuf, minIndexShuf, maxShuf, maxIndexShuf;
		for (int i = 1; i < WARP_SIZE; i *= 2) {

			minShuf = __shfl_xor_sync(FULL_MASK, min, i);
			minIndexShuf = __shfl_xor_sync(FULL_MASK, minIndex, i);
			if (min > minShuf) {
				minIndex = minIndexShuf;
				min = minShuf;
			}

			maxShuf = __shfl_xor_sync(FULL_MASK, max, i);
			maxIndexShuf = __shfl_xor_sync(FULL_MASK, maxIndex, i);
			if (max < maxShuf) {
				maxIndex = maxIndexShuf;
				max = maxShuf;
			}

		}

		if (lid == 0) {
			s_reducedMin[blockWid] = min;
			s_reducedMinIndex[blockWid] = minIndex;
			s_reducedMax[blockWid] = max;
			s_reducedMaxIndex[blockWid] = maxIndex;
		}

		__syncthreads();

		// 3. Warp reduction of reduced results
		if (blockWid == 0) {

			min = s_reducedMin[lid];
			minIndex = s_reducedMinIndex[lid];
			max = s_reducedMax[lid];
			maxIndex = s_reducedMaxIndex[lid];

			for (int i = 1; i < WARP_SIZE; i *= 2) {

				minShuf = __shfl_xor_sync(FULL_MASK, min, i);
				minIndexShuf = __shfl_xor_sync(FULL_MASK, minIndex, i);
				if (min > minShuf) {
					minIndex = minIndexShuf;
					min = minShuf;
				}

				maxShuf = __shfl_xor_sync(FULL_MASK, max, i);
				maxIndexShuf = __shfl_xor_sync(FULL_MASK, maxIndex, i);
				if (max < maxShuf) {
					maxIndex = maxIndexShuf;
					max = maxShuf;
				}

			}

			if (lid == 0) {
				if (minIndex == maxIndex) {
					islandBest[blockIdx.x] = 0;
					islandWorst[blockIdx.x] = islandPopulationSize - 1;
				} else {
					islandBest[blockIdx.x] = minIndex;
					islandWorst[blockIdx.x] = maxIndex;
				}
			}
		}
	}

	template <typename gene>
	__global__ void migrationKernel(gene* population, unsigned int islandPopulationSize, unsigned int nWarpSizeAligned, unsigned int *cycleWeight, unsigned int* islandBest, unsigned int* islandWorst, bool *sourceInSecondBuffer) {
		__shared__ gene *s_srcDst[2];
		__shared__ unsigned int s_reducedMax[WARP_SIZE];
		__shared__ unsigned int s_reducedMaxIndex[WARP_SIZE];

		unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;	// Global thread id
		unsigned int blockWid = (tid >> 5) & (WARP_SIZE - 1);		// Block warp id
		unsigned int lid = tid & (WARP_SIZE - 1);					// Warp thread id

		unsigned int thisIsland = blockIdx.x;
		unsigned int nextIsland = (blockIdx.x + 1) % gridDim.x;

		int minIndex; // Only set when threadIdx.x == 0
		if (threadIdx.x == 0) {
			unsigned int nextIslandBestWid = islandBest[nextIsland];
			unsigned int thisIslandBestWid = islandBest[thisIsland];
			unsigned int thisIslandWorstWid = islandWorst[thisIsland];

			unsigned int nextIslandBestWidGlobal = nextIsland * islandPopulationSize + nextIslandBestWid;
			unsigned int thisIslandBestWidGlobal = thisIsland * islandPopulationSize + thisIslandBestWid;
			unsigned int thisIslandWorstWidGlobal = thisIsland * islandPopulationSize + thisIslandWorstWid;

			unsigned int nextIslandBestCycleWeight = cycleWeight[nextIslandBestWidGlobal];
			unsigned int thisIslandBestCycleWeight = cycleWeight[thisIslandBestWidGlobal];

			cycleWeight[thisIslandWorstWidGlobal] = nextIslandBestCycleWeight;
			minIndex = thisIslandBestCycleWeight > nextIslandBestCycleWeight ? thisIslandWorstWid : thisIslandBestWid;

			bool nextSourceInSecondBuffer = sourceInSecondBuffer[nextIsland];
			bool thisSourceInSecondBuffer = sourceInSecondBuffer[thisIsland];

			s_srcDst[0] = population + (2 * nextIsland * islandPopulationSize + (nextSourceInSecondBuffer ? islandPopulationSize : 0) + nextIslandBestWid) * nWarpSizeAligned;
			s_srcDst[1] = population + (2 * thisIsland * islandPopulationSize + (thisSourceInSecondBuffer ? islandPopulationSize : 0) + thisIslandWorstWid) * nWarpSizeAligned;
		}

		__syncthreads();

		unsigned int max = 0, maxIndex = threadIdx.x;
		for (unsigned int chromosomeIndex = threadIdx.x; chromosomeIndex < islandPopulationSize; chromosomeIndex += blockDim.x) {
			unsigned int value = cycleWeight[thisIsland * islandPopulationSize + chromosomeIndex];
			if (max < value) {
				max = value;
				maxIndex = chromosomeIndex;
			}
		}

		unsigned int maxShuf, maxIndexShuf;
		for (int i = 1; i < WARP_SIZE; i *= 2) {
			maxShuf = __shfl_xor_sync(FULL_MASK, max, i);
			maxIndexShuf = __shfl_xor_sync(FULL_MASK, maxIndex, i);
			if (max < maxShuf) {
				maxIndex = maxIndexShuf;
				max = maxShuf;
			}
		}

		if (lid == 0) {
			s_reducedMax[blockWid] = max;
			s_reducedMaxIndex[blockWid] = maxIndex;
		}

		__syncthreads();

		if (blockWid == 0) {
			max = s_reducedMax[lid];
			maxIndex = s_reducedMaxIndex[lid];
			for (int i = 1; i < WARP_SIZE; i *= 2) {
				maxShuf = __shfl_xor_sync(FULL_MASK, max, i);
				maxIndexShuf = __shfl_xor_sync(FULL_MASK, maxIndex, i);
				if (max < maxShuf) {
					maxIndex = maxIndexShuf;
					max = maxShuf;
				}
			}

			if (lid == 0) {
				if (minIndex == maxIndex) {
					islandBest[thisIsland] = 0;
					islandWorst[thisIsland] = islandPopulationSize - 1;
				} else {
					islandBest[thisIsland] = minIndex;
					islandWorst[thisIsland] = maxIndex;
				}
			}
		}

		__syncthreads();

		// Replace worst chromosome in thisIsland with best chromosome from nextIsland
		gene* srcChromosome = s_srcDst[0];
		gene* dstChromosome = s_srcDst[1];

		for (unsigned int i = threadIdx.x; i < nWarpSizeAligned; i += blockDim.x)
			dstChromosome[i] = srcChromosome[i];
	}

	template <typename Instance, typename gene>
	__global__ void islandEvolutionKernel(const Instance instance, curandState* globalState, unsigned int iterationCount, gene* population, unsigned int islandPopulationSize, unsigned int* cycleWeight, unsigned int *islandBest, unsigned int *islandWorst, bool *sourceInSecondBuffer) {
		extern __shared__ unsigned int s_buffer[];
		unsigned int *s_reducedMin = s_buffer;
		unsigned int *s_reducedMinIndex = s_reducedMin + WARP_SIZE;
		unsigned int *s_reducedMax = s_reducedMinIndex + WARP_SIZE;
		unsigned int *s_reducedMaxIndex = s_reducedMax + WARP_SIZE;
		unsigned int *s_cycleWeight = s_buffer + 4 * WARP_SIZE;
		unsigned int* s_islandBestWorstIndex = s_cycleWeight + islandPopulationSize;
		float *s_roulletteWheelThreshold = (float*)(s_islandBestWorstIndex + 2);
		float *s_roulletteWheelThresholdSums = s_roulletteWheelThreshold + islandPopulationSize;

		unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;	// Global thread id
		unsigned int blockWid = threadIdx.x >> 5;					// Block warp id
		unsigned int lid = threadIdx.x & (WARP_SIZE - 1);			// Warp thread id
		unsigned int nWarpSizeAligned = (size(instance) & ~(WARP_SIZE - 1)) + WARP_SIZE;
		bool thisSourceInSecondBuffer = sourceInSecondBuffer[blockIdx.x];

		for (unsigned int chromosomeIndex = threadIdx.x; chromosomeIndex < islandPopulationSize; chromosomeIndex += blockDim.x)
			s_cycleWeight[chromosomeIndex] = cycleWeight[blockIdx.x * islandPopulationSize + chromosomeIndex];

		if (threadIdx.x == 0) {
			s_islandBestWorstIndex[0] = islandBest[blockIdx.x];
			s_islandBestWorstIndex[1] = islandWorst[blockIdx.x];
		}

		__syncthreads();

		while (iterationCount-- > 0) {

			// Selection
			{
				unsigned int min = s_cycleWeight[s_islandBestWorstIndex[0]];
				unsigned int max = s_cycleWeight[s_islandBestWorstIndex[1]];

				float intervalWidthSum = 0.0f;
				for (unsigned int baseOffset = blockWid * WARP_SIZE; baseOffset < islandPopulationSize; baseOffset += blockDim.x) {
					float intervalWidth = 0.0f, intervalWidthShfl;
					if (baseOffset + lid < islandPopulationSize)
						intervalWidth = min == max ? 1.0f : ((float)(max - s_cycleWeight[baseOffset + lid])) / ((float)(max - min));

					// Warp scan to get partial prefix sums
					for (unsigned int i = 1; i < WARP_SIZE; i *= 2) {
						intervalWidthShfl = __shfl_up_sync(FULL_MASK, intervalWidth, i);
						if (lid >= i) intervalWidth += intervalWidthShfl;
					}

					if (baseOffset + lid < islandPopulationSize)
						s_roulletteWheelThreshold[baseOffset + lid] = intervalWidth;

					if (lid == WARP_SIZE - 1)
						intervalWidthSum += intervalWidth;
				}

				if (lid == WARP_SIZE - 1) {
					s_roulletteWheelThresholdSums[blockWid] = intervalWidthSum;
				}

				__syncthreads();

				if (blockWid == 0) {
					intervalWidthSum = s_roulletteWheelThresholdSums[lid];
					for (unsigned int i = 1; i < WARP_SIZE; i *= 2) 
						intervalWidthSum += __shfl_xor_sync(FULL_MASK, intervalWidthSum, i);
					s_roulletteWheelThresholdSums[lid] = intervalWidthSum;
				}

				__syncthreads();

				float maxThreshold = s_roulletteWheelThresholdSums[0];

				for (unsigned int chromosomeIndex = blockWid; chromosomeIndex < islandPopulationSize; chromosomeIndex += (blockDim.x >> 5)) {

					// Select lane id based on roullette wheel selection
					float rnd;
					if (lid == 0) rnd = curand_uniform(globalState + tid) * maxThreshold;
					rnd = __shfl_sync(FULL_MASK, rnd, 0);

					unsigned int selectedIndex;
					for (unsigned int baseOffset = 0; baseOffset < islandPopulationSize; baseOffset += WARP_SIZE) {
						float threshold = baseOffset + lid < islandPopulationSize ? 
							s_roulletteWheelThreshold[baseOffset + lid] : s_roulletteWheelThreshold[islandPopulationSize - 1];
						unsigned int thresholdLid = __clz(~(__ballot_sync(FULL_MASK, rnd <= threshold)));
						if (thresholdLid) {
							selectedIndex = baseOffset + WARP_SIZE - thresholdLid;
							break;
						}
						rnd -= __shfl_sync(FULL_MASK, threshold, WARP_SIZE - 1);
					}

					// Copy selected chromosome
					gene* srcChromosome = population + nWarpSizeAligned *
						(blockIdx.x * 2 * islandPopulationSize + (thisSourceInSecondBuffer ? islandPopulationSize : 0) + selectedIndex);
					gene* dstChromosome = population + nWarpSizeAligned *
						(blockIdx.x * 2 * islandPopulationSize + (thisSourceInSecondBuffer ? 0 : islandPopulationSize) + chromosomeIndex);

					for (unsigned int i = lid; i < nWarpSizeAligned; i += WARP_SIZE)
						dstChromosome[i] = srcChromosome[i];
				}

				thisSourceInSecondBuffer = !thisSourceInSecondBuffer;
			}

			__syncthreads();

			// Crossover

			// Mutation

			// Fitness
			for (unsigned int chromosomeIndex = blockWid; chromosomeIndex < islandPopulationSize; chromosomeIndex += (blockDim.x >> 5)) {
				gene* chromosome = population + nWarpSizeAligned * 
					(blockIdx.x * 2 * islandPopulationSize + (thisSourceInSecondBuffer ? islandPopulationSize : 0) + chromosomeIndex);

				unsigned int thisCycleWeight = calculateCycleWeight(chromosome, instance);
				if (lid == 0) s_cycleWeight[chromosomeIndex] = thisCycleWeight;
			}

			__syncthreads();

			// Best and Worst
			// 1. Block stride loop thread reduction and output writing
			{
				unsigned int min = 0xffffffff, minIndex = threadIdx.x, max = 0, maxIndex = threadIdx.x;
				for (unsigned int chromosomeIndex = threadIdx.x; chromosomeIndex < islandPopulationSize; chromosomeIndex += blockDim.x) {
					unsigned int value = s_cycleWeight[chromosomeIndex];
					if (min > value) {
						min = value;
						minIndex = chromosomeIndex;
					}
					if (max < value) {
						max = value;
						maxIndex = chromosomeIndex;
					}
				}

				// 2. Warp reduction in each warp
				unsigned int minShuf, minIndexShuf, maxShuf, maxIndexShuf;
				for (int i = 1; i < WARP_SIZE; i *= 2) {

					minShuf = __shfl_xor_sync(FULL_MASK, min, i);
					minIndexShuf = __shfl_xor_sync(FULL_MASK, minIndex, i);
					if (min > minShuf) {
						minIndex = minIndexShuf;
						min = minShuf;
					}

					maxShuf = __shfl_xor_sync(FULL_MASK, max, i);
					maxIndexShuf = __shfl_xor_sync(FULL_MASK, maxIndex, i);
					if (max < maxShuf) {
						maxIndex = maxIndexShuf;
						max = maxShuf;
					}

				}

				if (lid == 0) {
					s_reducedMin[blockWid] = min;
					s_reducedMinIndex[blockWid] = minIndex;
					s_reducedMax[blockWid] = max;
					s_reducedMaxIndex[blockWid] = maxIndex;
				}

				__syncthreads();

				// 3. Warp reduction of reduced results
				if (blockWid == 0) {

					min = s_reducedMin[lid];
					minIndex = s_reducedMinIndex[lid];
					max = s_reducedMax[lid];
					maxIndex = s_reducedMaxIndex[lid];

					for (int i = 1; i < WARP_SIZE; i *= 2) {

						minShuf = __shfl_xor_sync(FULL_MASK, min, i);
						minIndexShuf = __shfl_xor_sync(FULL_MASK, minIndex, i);
						if (min > minShuf) {
							minIndex = minIndexShuf;
							min = minShuf;
						}

						maxShuf = __shfl_xor_sync(FULL_MASK, max, i);
						maxIndexShuf = __shfl_xor_sync(FULL_MASK, maxIndex, i);
						if (max < maxShuf) {
							maxIndex = maxIndexShuf;
							max = maxShuf;
						}

					}

					s_islandBestWorstIndex[0] = minIndex;
					s_islandBestWorstIndex[1] = maxIndex;
				}
			}

			__syncthreads();

			// If population is stable break;
		}

		for (unsigned int chromosomeIndex = threadIdx.x; chromosomeIndex < islandPopulationSize; chromosomeIndex += blockDim.x)
			cycleWeight[blockIdx.x * islandPopulationSize + chromosomeIndex] = s_cycleWeight[chromosomeIndex];

		if (threadIdx.x == 0) {
			if (s_islandBestWorstIndex[0] == s_islandBestWorstIndex[1]) {
				islandBest[blockIdx.x] = 0;
				islandWorst[blockIdx.x] = islandPopulationSize - 1;
			} else {
				islandBest[blockIdx.x] = s_islandBestWorstIndex[0];
				islandWorst[blockIdx.x] = s_islandBestWorstIndex[1];
			}
			sourceInSecondBuffer[blockIdx.x] = thisSourceInSecondBuffer;
		}
	}

	template <typename Instance, typename gene = unsigned short>
	int solveTSPFineGrained(const Instance instance, struct IslandGeneticAlgorithmOptions options, int blockWarpCount, int seed) {
		unsigned int n = size(instance);
		unsigned int nWarpSizeAligned = (n & ~(WARP_SIZE - 1)) + WARP_SIZE;
		unsigned int* d_cycleWeight, * d_islandBest, * d_islandWorst;
		bool *d_sourceInSecondBuffer;
		gene* d_population;
		curandState* d_globalState;
		unsigned int* h_cycleWeight = new unsigned int[options.islandCount * options.islandPopulationSize];
		unsigned int* h_islandBest = new unsigned int[options.islandCount];
		unsigned int* h_islandWorst = new unsigned int[options.islandCount];

		if (cudaMalloc(&d_cycleWeight, options.islandCount * options.islandPopulationSize * sizeof(unsigned int)) != cudaSuccess)
			return -1;

		if (cudaMalloc(&d_population, 2 * nWarpSizeAligned * options.islandCount * options.islandPopulationSize * sizeof(gene)) != cudaSuccess)
			return -1;

		if (cudaMalloc(&d_globalState, options.islandCount * options.islandPopulationSize * WARP_SIZE * sizeof(curandState)) != cudaSuccess)
			return -1;

		if (cudaMalloc(&d_islandBest, options.islandCount * sizeof(unsigned int)) != cudaSuccess)
			return -1;

		if (cudaMalloc(&d_islandWorst, options.islandCount * sizeof(unsigned int)) != cudaSuccess)
			return -1;

		if (cudaMalloc(&d_sourceInSecondBuffer, options.islandCount * sizeof(bool)) != cudaSuccess)
			return -1;

		setupCurand<<<options.islandCount, blockWarpCount * WARP_SIZE>>>(d_globalState, seed);

		initializationKernel<<<options.islandCount, blockWarpCount * WARP_SIZE, (options.islandPopulationSize + 4 * WARP_SIZE) * sizeof(unsigned int)>>>(
			instance, d_globalState, d_population, options.islandPopulationSize, d_cycleWeight, d_islandBest, d_islandWorst
		);

		if (cudaMemset(d_sourceInSecondBuffer, false, options.islandCount * sizeof(bool)) != cudaSuccess)
			return -1;

		if (cudaDeviceSynchronize() != cudaSuccess)
			return -1;

		if (cudaMemcpy(h_cycleWeight, d_cycleWeight, options.islandCount * options.islandPopulationSize * sizeof(unsigned int), cudaMemcpyDeviceToHost) != cudaSuccess)
			return -1;

		if (cudaMemcpy(h_islandBest, d_islandBest, options.islandCount * sizeof(unsigned int), cudaMemcpyDeviceToHost) != cudaSuccess)
			return -1;

		if (cudaMemcpy(h_islandWorst, d_islandWorst, options.islandCount * sizeof(unsigned int), cudaMemcpyDeviceToHost) != cudaSuccess)
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

			migrationKernel<<<options.islandCount, blockWarpCount * WARP_SIZE>>>(
				d_population, options.islandPopulationSize, nWarpSizeAligned, d_cycleWeight, d_islandBest, d_islandWorst, d_sourceInSecondBuffer
			);

			islandEvolutionKernel<<<options.islandCount, blockWarpCount * WARP_SIZE, (options.islandPopulationSize + 4 * WARP_SIZE + 2) * sizeof(unsigned int) + (options.islandPopulationSize + WARP_SIZE) * sizeof(float)>>>(
				instance, d_globalState, options.isolatedIterationCount, d_population, options.islandPopulationSize, d_cycleWeight, d_islandBest, d_islandWorst, d_sourceInSecondBuffer
			);

			if (cudaDeviceSynchronize() != cudaSuccess)
				return -1;

			if (cudaMemcpy(h_cycleWeight, d_cycleWeight, options.islandCount * options.islandPopulationSize * sizeof(unsigned int), cudaMemcpyDeviceToHost) != cudaSuccess)
				return -1;

			if (cudaMemcpy(h_islandBest, d_islandBest, options.islandCount * sizeof(unsigned int), cudaMemcpyDeviceToHost) != cudaSuccess)
				return -1;

			if (cudaMemcpy(h_islandWorst, d_islandWorst, options.islandCount * sizeof(unsigned int), cudaMemcpyDeviceToHost) != cudaSuccess)
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
		cudaFree(d_sourceInSecondBuffer);

		delete[] h_cycleWeight;
		delete[] h_islandBest;
		delete[] h_islandWorst;
		
		return 0;
	}

}

#endif __ALGORITHM_FINE_GRAINED_H__
